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
      "reconstruct": CommandReconstruct.init,
      "transformer": CommandTransformer.init,
      "tokenize": CommandTokenize.init,
      "dedup": CommandDedup.init,
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
    print("    \(CommandVQVAE.usage)")
    print("    \(CommandReconstruct.usage)")
    print("    \(CommandTokenize.usage)")
    print("    \(CommandTransformer.usage)")
    print("    \(CommandDedup.usage)")
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
