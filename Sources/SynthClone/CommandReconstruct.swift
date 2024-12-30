import Foundation
import Honeycrisp

class CommandReconstruct: Command {

  public static var usage: String {
    "reconstruct <model_path> <input_file> <output_file> [temperature (0.8)]"
  }

  let modelPath: String
  let inputPath: String
  let outputPath: String
  let temperature: Double

  let model: VQVAE

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if args.count != 3 && args.count != 4 {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }
    modelPath = args[0]
    inputPath = args[1]
    outputPath = args[2]
    if args.count == 4 {
      guard let temp = Double(args[3]) else {
        fatalError("invalid temperature: \(args[3])")
      }
      temperature = temp
    } else {
      temperature = 0.8
    }

    print("creating model...")
    model = CommandVQVAE.createModel()
  }

  override public func run() async throws {
    try await loadModel()
    let audio = try await readInput()
    let recon = reconstruct(audio)
    try await writeOutput(recon)
  }

  private func loadModel() async throws {
    print("loading from checkpoint: \(modelPath) ...")
    let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(CommandVQVAE.State.self, from: data)
    try model.loadState(state.model)
  }

  private func readInput() async throws -> Tensor {
    print("loading audio from \(inputPath) ...")
    guard
      let audio = try? loadAudio(
        path: inputPath, sampleCount: CommandVQVAE.sampleCount)
    else {
      fatalError("failed to load audio")
    }
    return audio
  }

  private func reconstruct(_ x: Tensor) -> Tensor {
    print("encoding and decoding audio ...")
    let input = x + Tensor(randnLike: x) * CommandVQVAE.inputNoise
    return Tensor.withGrad(enabled: false) {
      model.withMode(.inference) {
        model.sampleReconstruction(input.unsqueeze(axis: 0), temperature: Float(temperature))
          .squeeze(axis: 0)
      }
    }
  }

  private func writeOutput(_ x: Tensor) async throws {
    let data = try await tensorToAudio(
      tensor: x.printing(onForward: "writing to \(outputPath) ..."))
    try data.write(to: URL(filePath: outputPath))
  }

}
