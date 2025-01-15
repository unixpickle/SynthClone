import CryptoKit
import Foundation
import Honeycrisp

class CommandTokenize: Command {

  public static var usage: String {
    "tokenize <data_dir> <vq_path> <output_dir>"
  }

  public struct Shard: Codable {
    public struct Record: Codable {
      public let id: String
      public let caption: String
      public let tokens: [UInt16]
    }

    public var records: [Record]
  }

  let batchSize = 8

  let audioDir: URL
  let captionDir: URL
  let vqPath: String
  let outputDir: String
  let vqvae: VQVAE

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if args.count != 3 {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }
    let baseDir = URL(filePath: args[0])
    vqPath = args[1]
    outputDir = args[2]

    audioDir = maybeResolveSymlink(baseDir.appending(component: "outputs"))
    captionDir = maybeResolveSymlink(baseDir.appending(component: "inputs"))

    if !FileManager.default.fileExists(atPath: outputDir) {
      try FileManager.default.createDirectory(
        at: URL(filePath: outputDir), withIntermediateDirectories: false)
    }

    vqvae = VQVAE(channels: 1, vocab: 16384, latentChannels: 4, downsamples: 8)
  }

  override public func run() async throws {
    try loadVQ()
    let shards = try listShards()
    startFLOPCounter()
    for (shard, pathsAndCaptions) in shards.sorted(by: { x, y in x.key < y.key }) {
      try await tokenizeShard(shard: shard, pathsAndCaptions: pathsAndCaptions)
    }
  }

  func loadVQ() throws {
    print("loading VQVAE from checkpoint: \(vqPath) ...")
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(
      CommandVQVAE.State.self, from: try Data(contentsOf: URL(fileURLWithPath: vqPath)))
    try vqvae.loadState(state.model)
  }

  func listShards() throws -> [Int: [(URL, URL)]] {
    print("listing audio filenames...")
    var shards: [Int: [(URL, URL)]] = [:]
    let fileManager = FileManager.default
    var contents = try fileManager.contentsOfDirectory(
      at: audioDir,
      includingPropertiesForKeys: nil,
      options: []
    )
    contents.shuffle()
    var count = 0
    for audioURL in contents {
      if audioURL.pathExtension != "aiff" {
        continue
      }
      let idxStr = String(
        audioURL.lastPathComponent.split(separator: ".").first!.split(separator: "_").last!)
      let textURL = captionDir.appending(component: "line_\(idxStr).txt")
      if !fileManager.fileExists(atPath: textURL.path()) {
        continue
      }
      let shard = (Int(idxStr)! % 0x100)
      if shards[shard] == nil {
        shards[shard] = []
      }
      shards[shard]!.append((audioURL, textURL))
      count += 1
    }
    print(" => listed total of \(count) audios")
    return shards
  }

  func tokenizeShard(shard: Int, pathsAndCaptions: [(URL, URL)]) async throws {
    let fileManager = FileManager.default

    print("working on shard \(shard) ...")
    print(" => found \(pathsAndCaptions.count) audios in this shard")

    let shardURL = URL(filePath: outputDir).appending(component: "\(shard).plist")
    var shard: Shard = Shard(records: [])
    if fileManager.fileExists(atPath: shardURL.path()) {
      shard = try PropertyListDecoder().decode(Shard.self, from: try Data(contentsOf: shardURL))
      print(" => shard exists with \(shard.records.count) records")
    }

    let existingIDs = Set(shard.records.map { $0.id })
    var numFailed = 0
    var numSucceeded = 0

    var currentBatch: [Tensor] = []
    var currentIDs: [String] = []
    var currentCaptions: [String] = []

    func flushBatch() async throws {
      let vqs = Tensor.withGrad(enabled: false) {
        let audioTensor = Tensor(stack: currentBatch)
        let noise = Tensor(randnLike: audioTensor) * CommandVQVAE.inputNoise
        return vqvae.bottleneck(vqvae.encoder(audioTensor + noise)).codes
      }
      for (i, (id, caption)) in zip(currentIDs, currentCaptions).enumerated() {
        let tokens = try await vqs[i].ints().map { UInt16($0) }
        shard.records.append(
          Shard.Record(id: id, caption: caption, tokens: tokens))
      }
      currentBatch = []
      currentIDs = []
      currentCaptions = []
    }

    for (imagePath, captionPath) in pathsAndCaptions {
      let imageID = imagePath.lastPathComponent
      if existingIDs.contains(imageID) {
        continue
      }
      do {
        let caption = String(decoding: try Data(contentsOf: captionPath), as: UTF8.self)
        guard
          let audio = try? loadAudio(path: imagePath.path(), sampleCount: CommandVQVAE.sampleCount)
        else {
          numFailed += 1
          continue
        }
        currentBatch.append(audio)
        currentIDs.append(imageID)
        currentCaptions.append(caption)
        if currentBatch.count == batchSize {
          try await flushBatch()
        }
        numSucceeded += 1
      } catch {
        print("\(error)")
        numFailed += 1
      }
    }
    if !currentBatch.isEmpty {
      try await flushBatch()
    }
    let data = try PropertyListEncoder().encode(shard)
    try data.write(to: shardURL, options: .atomic)
    print(" => added \(numSucceeded) records successfully with \(numFailed) errors")
  }

}

func maybeResolveSymlink(_ path: URL) -> URL {
  do {
    let resolvedPath = try FileManager.default.destinationOfSymbolicLink(atPath: path.path())
    let absoluteResolvedPath =
      (resolvedPath as NSString).isAbsolutePath
      ? resolvedPath
      : (path.path() as NSString).deletingLastPathComponent + "/" + resolvedPath
    return URL(filePath: absoluteResolvedPath)
  } catch {
    return path
  }
}
