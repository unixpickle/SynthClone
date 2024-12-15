import AVFoundation
import Honeycrisp

enum AudioError: Error {
  case createConverter
  case createPCMBuffer
  case emptyAudio
}

func loadAudio(path: String, sampleCount: Int, sampleRate: Int = 24000) throws -> Tensor {
  let audioFile = try AVAudioFile(forReading: URL(filePath: path))

  let format = audioFile.processingFormat
  let outputFormat = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
  guard let converter = AVAudioConverter(from: format, to: outputFormat) else {
    throw AudioError.createConverter
  }

  let inputBuffer = AVAudioPCMBuffer(
    pcmFormat: format,
    frameCapacity: AVAudioFrameCount(audioFile.length)
  )!
  try audioFile.read(into: inputBuffer)

  let convertedBuffer = AVAudioPCMBuffer(
    pcmFormat: outputFormat,
    frameCapacity: AVAudioFrameCount(
      UInt32(Double(audioFile.length) * Double(sampleRate) / format.sampleRate))
  )!

  var error: NSError?
  let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
    outStatus.pointee = .haveData
    return inputBuffer
  }

  converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)
  if let error = error {
    throw error
  }

  let audioData = convertedBuffer.floatChannelData![0]
  let numberOfFrames = Int(convertedBuffer.frameLength)

  var rawAudioSamples: [Float] = Array(UnsafeBufferPointer(start: audioData, count: numberOfFrames))
  while rawAudioSamples.count < sampleCount {
    rawAudioSamples.append(0)
  }
  rawAudioSamples = Array(rawAudioSamples.prefix(sampleCount))

  if (rawAudioSamples.max() ?? 0) == 0 {
    throw AudioError.emptyAudio
  }

  return compressMlaw(pcm: Tensor(data: rawAudioSamples, shape: [1, rawAudioSamples.count]))
}

func tensorToAudio(tensor: Tensor, sampleRate: Int = 24000) async throws -> Data {
  let rawAudioSamples = try await invertMlaw(mlaw: tensor).floats()

  let audioFormat = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
  let frameCount = AVAudioFrameCount(rawAudioSamples.count)
  guard let audioBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
    throw AudioError.createPCMBuffer
  }
  audioBuffer.frameLength = frameCount

  let audioData = audioBuffer.floatChannelData![0]
  for i in 0..<rawAudioSamples.count {
    audioData[i] = rawAudioSamples[i]
  }

  let tempFileURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    .appendingPathExtension("wav")
  defer {
    try? FileManager.default.removeItem(at: tempFileURL)
  }

  // Use a scope to make sure file is closed and header
  // is written before re-reading.
  try {
    let audioFile = try AVAudioFile(forWriting: tempFileURL, settings: audioFormat.settings)
    try audioFile.write(from: audioBuffer)
  }()

  return try Data(contentsOf: tempFileURL)
}

private func compressMlaw(pcm: Tensor) -> Tensor {
  let sign = (pcm > 0).when(isTrue: Tensor(onesLike: pcm), isFalse: -1.0)
  let abs = pcm.abs()
  // 5.5451774445 = ln(1 + 255)
  return sign * (1 + 255 * abs).log() / 5.5451774445
}

private func invertMlaw(mlaw: Tensor) -> Tensor {
  let sign = (mlaw > 0).when(isTrue: Tensor(onesLike: mlaw), isFalse: -1.0)
  let abs = mlaw.abs()
  // 5.5451774445 = ln(1 + 255)
  return sign * ((5.5451774445 * abs).exp() - 1) / 255
}
