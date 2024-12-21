import Foundation
import Honeycrisp

class CommandVQVAE: Command {

  public struct State: Codable {
    let step: Int
    let model: Trainable.State
    let dataset: AudioDataLoader.State?
    let opt: Adam.State?
  }

  //let sampleCount: Int = 1024 * 12 * 5
  let sampleCount: Int = 32768

  let lr: Float = 0.00001
  let bs = 2
  let reviveInterval = 500
  let reviveBatches = 16
  let commitCoeff = 0.1
  let inputNoise = 0.001
  let gradClip = 1.0

  let savePath: String
  let samplePath: String
  let audioDir: String
  let model: VQVAE
  let opt: Adam
  var step: Int = 0
  var dataStream: AsyncThrowingStream<(Tensor, AudioDataLoader.State), Error>?

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if args.count != 3 {
      print("Usage: ... vqvae <audio_dir> <sample_path> <save_path>")
      throw ArgumentError.invalidArgs
    }
    audioDir = args[0]
    samplePath = args[1]
    savePath = args[2]

    print("creating model and optimizer...")
    model = VQVAE(channels: 1, vocab: 16384, latentChannels: 4, downsamples: 1)
    opt = Adam(model.parameters, lr: lr, weightDecay: 0.1)
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
      batchSize: bs, audios: try AudioIterator(audioDir: audioDir, sampleCount: sampleCount))
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
    return
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
      let batch = batch + Tensor(randnLike: batch) * inputNoise
      step += 1
      let nll = model(batch)
      let loss = nll.mean()
      (loss*0).backward()
      let gradNorm = clipGradients(model: model, threshold: gradClip)
      opt.step()
      opt.clearGrads()
      print(
        "step \(step):"
          + " loss=\(try await loss.item())"
          + " grad_norm=\(try await gradNorm.item())"
          + " gflops=\(gflops)")
    }
  }

  private func sampleAndSave() async throws {
    print("dumping samples to: \(samplePath) ...")
    var it = dataStream!.makeAsyncIterator()
    let (inputRaw, dataState) = try await it.next()!
    let input = inputRaw + Tensor(randnLike: inputRaw) * inputNoise
    let output = Tensor.withGrad(enabled: false) {
      model.withMode(.inference) {
        model.sampleReconstruction(input)
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
      opt: try await opt.state()
    )
    let stateData = try PropertyListEncoder().encode(state)
    try stateData.write(to: URL(filePath: savePath), options: .atomic)
  }

}
