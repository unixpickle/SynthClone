import Foundation
import Honeycrisp

struct TrainAndEval<T> {
  let train: T
  let eval: T
}

extension TrainAndEval: Codable where T: Codable {}
extension TrainAndEval: Sendable where T: Sendable {}

class CommandTransformer: Command {

  public static var usage: String {
    "transformer <data_dir> <vq_path> <save_path>"
  }

  typealias DataStream = AsyncThrowingStream<
    TrainAndEval<(Tensor, CaptionedSequenceDataLoader.State)>, Error
  >

  public struct State: Codable {
    let step: Int
    let model: Trainable.State
    let dataset: TrainAndEval<CaptionedSequenceDataLoader.State>?
    let opt: Adam.State?
  }

  let testCaptions = [
    "The quick brown fox jumps over the lazy dog",
    "testing 1 2 3",
    "Hello, World!",
    "Alex said to Samantha.",
  ]

  let sampleFilename = "samples.aiff"

  let lr: Float = 0.0001
  let bs = 8
  let microbatch = 2
  let captionBytes: Int = 128
  let vqCount: Int = CommandVQVAE.sampleCount >> 8
  let saveInterval: Int = 5000
  let cfgProb: Float = 0.1
  let cfgScales: [Float] = [1.0, 2.0]

  let savePath: String
  let vqPath: String
  let dataDir: String
  let vqvae: VQVAE
  let model: Transformer
  let opt: Adam
  var step: Int = 0

  var dataStream: DataStream?
  var captionTensor: (@Sendable (String) -> Tensor)?
  var lastDataState: TrainAndEval<CaptionedSequenceDataLoader.State>?

  override internal var flopCount: Int64 {
    return super.flopCount
  }

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .heap(13_000_000_000))

    if args.count != 3 {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }
    dataDir = args[0]
    vqPath = args[1]
    savePath = args[2]

    vqvae = CommandVQVAE.createModel()
    model = Transformer(
      config: TransformerConfig(
        VocabSize: vqvae.bottleneck.vocab + 256, TokenCount: captionBytes + vqCount))
    opt = Adam(model.parameters, lr: lr)
  }

  override public func run() async throws {
    try await prepare()

    while true {
      try await trainInnerLoop()
      try await sampleAndSave()
    }
  }

  private func prepare() async throws {
    let dataLoader = TrainAndEval(
      train: try CaptionedSequenceDataLoader(
        batchSize: bs, dropProb: cfgProb, captionLength: captionBytes,
        captionTokenOffset: vqvae.bottleneck.vocab, shardDir: dataDir),
      eval: try CaptionedSequenceDataLoader(
        batchSize: bs, dropProb: cfgProb, captionLength: captionBytes,
        captionTokenOffset: vqvae.bottleneck.vocab, shardDir: dataDir, isEval: true)
    )
    self.captionTensor = dataLoader.train.captionTensor

    print("loading VQVAE from checkpoint: \(vqPath) ...")
    let data = try Data(contentsOf: URL(fileURLWithPath: vqPath))
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(CommandVQVAE.State.self, from: data)
    try vqvae.loadState(state.model)

    if FileManager.default.fileExists(atPath: savePath) {
      print("loading from checkpoint: \(savePath) ...")
      let data = try Data(contentsOf: URL(fileURLWithPath: savePath))
      let decoder = PropertyListDecoder()
      let state = try decoder.decode(State.self, from: data)
      try model.loadState(state.model)
      if let optState = state.opt {
        try opt.loadState(optState)
      }
      if let dataState = state.dataset {
        dataLoader.train.state = dataState.train
        dataLoader.eval.state = dataState.eval
      }
      step = state.step
    }

    let it = zip(dataLoader.train, dataLoader.eval).lazy.map {
      x, y -> Result<TrainAndEval<(Tensor, CaptionedSequenceDataLoader.State)>, Error> in
      switch x {
      case .failure(let e):
        return .failure(e)
      case .success(let x):
        switch y {
        case .failure(let e):
          return .failure(e)
        case .success(let y):
          return .success(TrainAndEval(train: x, eval: y))
        }
      }
    }
    dataStream = loadDataInBackgroundSending(it)
  }

  private func captionTensor(_ captions: [String]) -> Tensor {
    Tensor(stack: captions.map(captionTensor!))
  }

  private func takeDataset(_ n: Int) -> AsyncPrefixSequence<DataStream> {
    dataStream!.prefix(n)
  }

  private func trainInnerLoop() async throws {
    func loss(_ batch: Tensor) -> Tensor {
      // We do not model the caption prefix, only the VQ tokens.
      // If we truncate the input, the dimensions aren't aligned well and slow
      // down training significantly.
      //     let outputs = model(batch[..., ..<(-1)])[..., (captionBytes - 1)...]
      let outputs = model(batch)[..., (captionBytes - 1)..<(batch.shape[1] - 1)]
      let targets = batch[..., captionBytes...]

      let logProbs = outputs.logSoftmax(axis: -1)
      let losses = -logProbs.gather(axis: -1, indices: targets.unsqueeze(axis: -1))
      return losses.mean()
    }

    print("training...")
    for try await batch in takeDataset(saveInterval) {
      lastDataState = TrainAndEval(train: batch.train.1, eval: batch.eval.1)
      step += 1

      let evalLoss = Tensor.withGrad(enabled: false) { loss(batch.eval.0) }
      try await evalLoss.wait()

      var trainLosses = [Tensor]()
      for i in stride(from: 0, to: bs, by: microbatch) {
        let smallBatch = min(bs - i, microbatch)
        let scale = Float(smallBatch) / Float(bs)
        let trainLoss = loss(batch.train.0[i..<(i + smallBatch)])
        (trainLoss * scale).backward()
        trainLosses.append(trainLoss.noGrad() * scale)

        // Ensure that the memory from the step is done being used.
        for (_, p) in model.parameters {
          try await p.grad?.wait()
        }
      }
      let trainLoss = trainLosses.reduce(Tensor(data: [0.0], shape: []), { x, y in x + y })
      let gradNorm = try await model.gradNorm()
      if !gradNorm.isFinite {
        fatalError("got NaN gradient")
      }
      opt.step()
      opt.clearGrads()
      print(
        "step \(step):"
          + " loss=\(try await trainLoss.item())"
          + " valid_loss=\(try await evalLoss.item())"
          + " grad_norm=\(gradNorm)"
          + " gflops=\(gflops)")
    }
  }

  private func sampleAndSave() async throws {
    print("sampling to \(sampleFilename) ...")
    var waveforms = [Tensor]()
    for scale in cfgScales {
      let gen = Backend.current.createRandom()
      gen.seed(step)

      let captions = captionTensor(testCaptions)
      let samples = try await model.sample(
        prefixes: captions, generator: gen, cfgScale: scale, logInterval: 10)

      // Make sure we have the tokens before using
      // memory for decoder.
      try await samples.wait()

      // The decoder is memory hungry, so we microbatch
      // with batch size 1.
      for i in 0..<samples.shape[0] {
        try await Tensor.withGrad(enabled: false) {
          let waveform = vqvae.sampleFromVQ(samples[i, NewAxis()]).squeeze(axis: 0)
          try await waveform.wait()
          waveforms.append(waveform)
        }
      }
    }
    let audio = try await tensorToAudio(tensor: Tensor(concat: waveforms, axis: 1))
    try audio.write(to: URL(filePath: sampleFilename))

    print("saving to \(savePath) ...")
    let state = State(
      step: step,
      model: try await model.state(),
      dataset: lastDataState!,
      opt: try await opt.state()
    )
    let stateData = try PropertyListEncoder().encode(state)
    try stateData.write(to: URL(filePath: savePath), options: .atomic)
  }

}
