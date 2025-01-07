import Foundation
import Honeycrisp

class CommandVQVAE: Command {

  public static var usage: String {
    "vqvae <audio_dir> <sample_path> <save_path>"
  }

  public struct State: Codable {
    let step: Int
    let model: Trainable.State
    let dataset: AudioDataLoader.State?
    let opt: Adam.State?
    let clipper: GradClipper.State?
  }

  static let sampleCount: Int = 1024 * 24 * 5
  static let inputNoise = 0.0001

  let lr: Float = 0.00005
  let bs = 2
  let reviveInterval = 500
  let reviveBatches = 16
  let commitCoeff = 5.0
  let sampleTemp = 0.8

  let savePath: String
  let samplePath: String
  let audioDir: String
  let model: VQVAE
  let opt: Adam
  let clipper: GradClipper
  var step: Int = 0
  var dataStream: AsyncThrowingStream<(Tensor, AudioDataLoader.State), Error>?

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if args.count != 3 {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }
    audioDir = args[0]
    samplePath = args[1]
    savePath = args[2]

    print("creating model and optimizer...")
    model = Self.createModel()
    opt = Adam(model.parameters, lr: lr)
    clipper = GradClipper()
  }

  static public func createModel() -> VQVAE {
    VQVAE(channels: 1, vocab: 16384, latentChannels: 4, downsamples: 8)
  }

  override public func run() async throws {
    try await prepare()

    while true {
      try await revive()
      try await trainInnerLoop()
      try await sampleAndSave()
    }
  }

  private func prepare() async throws {
    print("creating data loader...")
    let dataLoader = AudioDataLoader(
      batchSize: bs, audios: try AudioIterator(audioDir: audioDir, sampleCount: Self.sampleCount))
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
        dataLoader.state = dataState
      }
      if let clipState = state.clipper {
        clipper.state = clipState
      }
      step = state.step
    }
    dataStream = loadDataInBackgroundSending(dataLoader)
  }

  private func takeDataset(_ n: Int) -> AsyncPrefixSequence<
    AsyncThrowingStream<(Tensor, AudioDataLoader.State), Error>
  > {
    return dataStream!.prefix(n)
  }

  private func revive() async throws {
    print("reviving unused dictionary entries...")
    print(" => collecting features...")
    var reviveBatch = [Tensor]()
    for try await (x, _) in takeDataset(reviveBatches) {
      reviveBatch.append(x)
    }
    let revivedCount = Tensor.withGrad(enabled: false) {
      let features = model.withMode(.inference) {
        Tensor(concat: reviveBatch.map(model.features))
      }
      print(" => collected \(features.shape[0]) features")
      return model.bottleneck.revive(features)
    }
    print(" => revived \(try await revivedCount.ints()[0]) entries")
  }

  private func trainInnerLoop() async throws {
    print("training...")
    for try await (batch, _) in takeDataset(reviveInterval) {
      let batch = batch + Tensor(randnLike: batch) * Self.inputNoise
      step += 1
      let (nll, vqLosses) = model(batch)
      let loss = nll.mean()
      (loss + vqLosses.codebookLoss + commitCoeff * vqLosses.commitmentLoss).backward()
      let (gradNorm, clipScale) = try await clipper.clipGrads(model: model)
      opt.step()
      opt.clearGrads()
      print(
        "step \(step):"
          + " loss=\(try await loss.item())"
          + " commitment=\(try await vqLosses.commitmentLoss.item())"
          + " grad_norm=\(gradNorm)"
          + " grad_scale=\(clipScale)"
          + " gflops=\(gflops)")
    }
  }

  private func sampleAndSave() async throws {
    print("dumping samples to: \(samplePath) ...")
    var it = dataStream!.makeAsyncIterator()
    let (inputRaw, dataState) = try await it.next()!
    let input = inputRaw + Tensor(randnLike: inputRaw) * Self.inputNoise
    let output = Tensor.withGrad(enabled: false) {
      model.withMode(.inference) {
        model.sampleReconstruction(input, temperature: Float(sampleTemp))
      }
    }
    let audios = Tensor(concat: [input, output], axis: -1)
    let data = try await tensorToAudio(tensor: audios.move(axis: 1, to: -1).flatten(endAxis: 1))
    try data.write(to: URL(filePath: samplePath))

    print("saving to \(savePath) ...")
    let state = State(
      step: step,
      model: try await model.state(),
      dataset: dataState,
      opt: try await opt.state(),
      clipper: clipper.state
    )
    let stateData = try PropertyListEncoder().encode(state)
    try stateData.write(to: URL(filePath: savePath), options: .atomic)
  }

}
