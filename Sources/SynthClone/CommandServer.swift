import Foundation
import Honeycrisp
import Vapor

enum ServerError: Error {
  case invalidPort(String)
  case missingResource(String)
  case loadResource(String)
}

class CommandServer: Command {

  public static var usage: String {
    "server <vqvae_path> <transformer_path> [port]"
  }

  let loadPath: String
  let vqPath: String
  let port: Int

  let vqvae: SyncTrainable<VQVAE>
  let model: SyncTrainable<Transformer>

  let captionBytes: Int = CommandTransformer.captionBytes
  var captionTokenOffset: Int {
    vqvae.use { vqvae in vqvae.bottleneck.vocab }
  }

  init(_ args: [String]) throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if ![2, 3].contains(args.count) {
      print("Usage: ... \(Self.usage)")
      throw ArgumentError.invalidArgs
    }
    vqPath = args[0]
    loadPath = args[1]
    guard let port = args.count > 2 ? Int(args[2]) : 8080 else {
      throw ServerError.invalidPort(args[2])
    }
    self.port = port

    vqvae = SyncTrainable(CommandVQVAE.createModel())
    model = SyncTrainable(
      Transformer(
        config: TransformerConfig(
          VocabSize: vqvae.use { vqvae in vqvae.bottleneck.vocab } + 256,
          TokenCount: captionBytes + CommandTransformer.vqCount
        )
      )
    )
  }

  override public func run() async throws {
    try loadVQVAE()
    try loadModel()

    let app = try await Application.make(.detect(arguments: ["serve"]))
    app.http.server.configuration.hostname = "0.0.0.0"
    app.http.server.configuration.port = port

    let filenames = ["index.html", "app.js"]
    let contentTypes = ["html": "text/html", "js": "text/javascript"]
    for filename in filenames {
      let parts = filename.split(separator: ".")
      guard
        let url = Bundle.module.url(forResource: String(parts[0]), withExtension: String(parts[1]))
      else {
        throw ServerError.missingResource(filename)
      }
      guard let contents = try? Data(contentsOf: url) else {
        throw ServerError.loadResource(filename)
      }
      app.on(.GET, filename == "index.html" ? "" : "\(filename)") { request -> Response in
        Response(
          status: .ok,
          headers: ["content-type": contentTypes[String(parts[1])]!],
          body: .init(data: contents))
      }
    }

    let sampleFunc = sample
    app.responseCompression(.disable, force: true).on(.GET, "sample") { request -> Response in
      guard let prompt = request.query[String.self, at: "prompt"] else {
        print("missing prompt in query")
        return Response(status: .badRequest)
      }

      guard let guidanceScale = request.query[Float.self, at: "guidanceScale"] else {
        print("missing guidanceScale in query")
        return Response(status: .badRequest)
      }

      let response = Response(
        body: Response.Body.init(asyncStream: { writer in
          print("starting sampling for prompt: \(prompt)")
          defer { print("sampling request exiting for prompt: \(prompt)") }

          // This seems to work around server-side buffering that would
          // otherwise delay the response.
          let bufferFiller = [String](repeating: "\n", count: 4096).joined().utf8
          try await writer.write(.buffer(ByteBuffer(data: Data(bufferFiller))))

          for try await token in sampleFunc(prompt, guidanceScale) {
            try await writer.write(.buffer(ByteBuffer(bytes: Data("\(token)\n".utf8))))
          }
          try await writer.write(.end)
        })
      )

      return response
    }

    let vqvae = vqvae
    let decodeLock = NSLock()
    app.on(.GET, "decode") { request -> Response in
      // To save memory, we will only decode one VQ sequence at once.
      let _: () = await withCheckedContinuation { continuation in
        DispatchQueue.global().sync {
          decodeLock.lock()
          continuation.resume(returning: ())
        }
      }
      defer {
        DispatchQueue.global().sync { decodeLock.unlock() }
      }

      guard let tokenStr = request.query[String.self, at: "tokens"] else {
        return Response(status: .badRequest)
      }
      var tokens = [Int]()
      for comp in tokenStr.split(separator: ",") {
        guard let token = Int(comp) else {
          return Response(status: .badRequest)
        }
        tokens.append(token)
      }
      if tokens.count > CommandTransformer.vqCount {
        return Response(status: .badRequest)
      }

      while tokens.count < CommandTransformer.vqCount {
        tokens.append(tokens.last ?? 0)
      }

      let decoded = Tensor.withGrad(enabled: false) {
        vqvae.use {
          $0.sampleFromVQ(Tensor(data: tokens).unsqueeze(axis: 0)).squeeze(axis: 0)
        }
      }

      let audio = try await tensorToAudio(tensor: decoded)
      return Response(
        status: .ok,
        headers: ["content-type": "audio/wav"],
        body: .init(data: audio)
      )
    }

    try await app.execute()
  }

  private func loadVQVAE() throws {
    print("loading VQVAE from checkpoint: \(vqPath) ...")
    let data = try Data(contentsOf: URL(fileURLWithPath: vqPath))
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(CommandVQVAE.State.self, from: data)
    try vqvae.use { vqvae in try vqvae.loadState(state.model) }
  }

  private func loadModel() throws {
    print("loading model from checkpoint: \(loadPath) ...")
    let data = try Data(contentsOf: URL(fileURLWithPath: loadPath))
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(CommandTransformer.State.self, from: data)
    try model.use { model in try model.loadState(state.model) }
  }

  var captionTensor: @Sendable (String) -> Tensor {
    let captionBytes = captionBytes
    let captionTokenOffset = captionTokenOffset
    return { (caption: String) in
      var textTokens = [Int](repeating: 0, count: captionBytes)
      for (j, char) in caption.utf8.enumerated() {
        if j >= captionBytes {
          break
        }
        textTokens[j] = Int(char) + captionTokenOffset
      }
      return Tensor(data: textTokens, shape: [1, captionBytes])
    }
  }

  private var sample: @Sendable (String, Float) -> AsyncThrowingStream<Int, Error> {
    let model = model
    let captionTensor = captionTensor
    return { (prompt: String, cfgScale: Float) -> AsyncThrowingStream<Int, Error> in
      AsyncThrowingStream { continuation in
        let t = Task {
          do {
            for await x in model.use({ model in
              model.sampleStream(prefixes: captionTensor(prompt), cfgScale: cfgScale)
            }) {
              if Task.isCancelled {
                return
              }
              do {
                continuation.yield(try await x.ints()[0])
              } catch {
                continuation.finish(throwing: error)
                return
              }
            }
          }
          continuation.finish()
        }
        continuation.onTermination = { _ in t.cancel() }
      }
    }
  }

}
