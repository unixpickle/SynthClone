import Cocoa
import Foundation
import Honeycrisp

@main
struct Main {
  static func main() async {
    if CommandLine.arguments.count < 2 {
      printHelp()
      return
    }
    let commands = [
      "vqvae": CommandVQVAE.init,
      "transformer": CommandTransformer.init,
      "tokenize": CommandTokenize.init,
    ]
    guard let command = commands[CommandLine.arguments[1]] else {
      print("Unrecognized subcommand: \(CommandLine.arguments[1])")
      printHelp()
      return
    }
    do {
      let runner = try command(Array(CommandLine.arguments[2...]))
      try await runner.run()
    } catch {
      print("ERROR: \(error)")
    }
  }

  static func printHelp() {
    print("Usage: SynthClone <subcommand> ...")
    print("Subcommands:")
    print("    vqvae <data_dir> <sample_path> <state_path>")
    print("    tokenize <data_dir> <vqvae_path> <tok_dir>")
    print("    transformer <tok_dir> <vqvae_path> <state_path>")
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
