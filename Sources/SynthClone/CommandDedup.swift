import Foundation
import Honeycrisp

enum IterateDirError: Error {
  case createEnumerator
}

class CommandDedup: Command {

  public static var usage: String {
    "dedup <input_caption_dir> <output_caption_dir>"
  }

  let inputDir: String
  let outputDir: String

  init(_ args: [String]) throws {
    if args.count != 2 {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }

    inputDir = args[0]
    outputDir = args[1]
  }

  override public func run() async throws {
    print("listing input files...")
    var i = 0
    var lastPrint = DispatchTime.now().uptimeNanoseconds
    var allContents = [Data: URL]()

    func printStatus() {
      print("\renumerated \(i) paths (with \(allContents.count) unique)", terminator: "")
      fflush(stdout)
    }

    for try await fileURL in iterateInputPaths() {
      allContents[try Data(contentsOf: fileURL)] = fileURL
      let now = DispatchTime.now().uptimeNanoseconds
      if i == 0 || now > lastPrint + 1_000_000_000 {
        printStatus()
        lastPrint = now
      }
      i += 1
    }
    printStatus()
    print("")

    print("writing files to \(outputDir)...")
    let baseURL = URL(filePath: outputDir)
    for (content, inURL) in allContents {
      let outURL = baseURL.appending(component: inURL.pathComponents.last!)
      try content.write(to: outURL)
    }
  }

  func iterateInputPaths() -> AsyncThrowingStream<URL, Error> {
    let inputDir = inputDir
    return AsyncThrowingStream { continuation in
      DispatchQueue.global().async {
        let baseURL = URL(filePath: inputDir)
        var gotError = false
        guard
          let enumerator = FileManager.default.enumerator(
            at: baseURL,
            includingPropertiesForKeys: nil,
            options: [.skipsSubdirectoryDescendants],
            errorHandler: { _, err in
              gotError = true
              continuation.finish(throwing: err)
              return false
            }
          )
        else {
          continuation.finish(throwing: IterateDirError.createEnumerator)
          return
        }
        for case let fileURL as URL in enumerator {
          if gotError {
            preconditionFailure("error block should have stopped enumerator")
          }
          continuation.yield(fileURL)
        }
        if !gotError {
          continuation.finish()
        }
      }
    }
  }

}
