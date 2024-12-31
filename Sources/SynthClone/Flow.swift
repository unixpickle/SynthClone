import Foundation
import HCBacktrace
import Honeycrisp

class FlowLayer: Trainable {
  let isEven: Bool

  @Child var condConv: Conv1D
  @Child var conv1: Conv1D
  @Child var conv2: Conv1D
  @Child var conv3: Conv1D
  @Child var norm: GroupNorm

  init(isEven: Bool, condChannels: Int, hiddenChannels: Int) {
    self.isEven = isEven
    super.init()
    condConv = Conv1D(
      inChannels: condChannels, outChannels: hiddenChannels, kernelSize: 5, stride: 2,
      padding: .allSides(2))
    conv1 = Conv1D(
      inChannels: 1, outChannels: hiddenChannels, kernelSize: 5, stride: 1,
      padding: .same)
    conv2 = Conv1D(
      inChannels: hiddenChannels, outChannels: hiddenChannels, kernelSize: 5, stride: 1,
      padding: .same)
    conv3 = Conv1D(
      inChannels: hiddenChannels, outChannels: 2, kernelSize: 5, stride: 1, padding: .same)
    norm = GroupNorm(groupCount: 32, channelCount: hiddenChannels)
    for (_, var p) in conv3.parameters {
      p.data! = Tensor(zerosLike: p.data!)
    }
  }

  func noiseToSample(noise: Tensor, cond: Tensor) -> Tensor {
    applyModel(noise, cond: cond).out
  }

  func sampleToNoise(sample: Tensor, cond: Tensor) -> (out: Tensor, logScale: Tensor) {
    return applyModel(sample, cond: cond, invert: true)
  }

  private func applyModel(_ x: Tensor, cond: Tensor, invert: Bool = false) -> (
    out: Tensor, logScale: Tensor
  ) {
    #alwaysAssert(x.shape.count == 3)
    #alwaysAssert(x.shape[1] == 1)
    #alwaysAssert(x.shape[2] % 2 == 0)

    let split = splitAlongTime(x)
    let (inputs, toModify) =
      if isEven {
        split
      } else {
        (split.1, split.0)
      }

    var h = conv1(inputs) + condConv(cond)
    h = norm(h)
    h = h.gelu()
    h = conv2(h)
    h = h.gelu()
    h = conv3(h)
    let scaleAndBias = h.chunk(axis: 1, count: 2)
    let (scaleParams, bias) = (scaleAndBias[0], scaleAndBias[1])

    let logScale = scaleParams.atan()
    let scale = logScale.exp()

    let modified =
      if invert {
        (toModify - bias) / scale
      } else {
        toModify.mul(scale, thenAdd: bias)
      }

    let combinedOut =
      if isEven {
        unsplitAlongTime(inputs, modified)
      } else {
        unsplitAlongTime(modified, inputs)
      }

    return (out: combinedOut, logScale: logScale.flatten(startAxis: 1).sum(axis: 1))
  }
}

class FlowModel: Trainable {
  let layersPerResolution: Int
  @Child var layers: TrainableArray<FlowLayer>
  @Child var condDownsample: TrainableArray<Conv1D>

  init(
    condChannels: Int,
    downsamples: Int = 2,
    layersPerResolution: Int = 12,
    hiddenChannels: Int = 192
  ) {
    self.layersPerResolution = layersPerResolution
    super.init()
    layers = TrainableArray(
      (0..<(downsamples * layersPerResolution)).map { i in
        FlowLayer(
          isEven: i % 2 == 0,
          condChannels: condChannels,
          hiddenChannels: hiddenChannels
        )
      })
    condDownsample = TrainableArray(
      (0..<(downsamples - 1)).map { _ in
        Conv1D(
          inChannels: condChannels, outChannels: condChannels, kernelSize: 5, stride: 2,
          padding: .allSides(2))
      })
  }

  func noiseToSample(cond: Tensor, temperature: Float = 1.0) -> Tensor {
    let conds: [Tensor] = computeConds(cond: cond).reversed()
    var h = Tensor(randn: [cond.shape[0], 1, conds.first!.shape[2]]) * temperature
    for (i, layer) in layers.children.enumerated() {
      let condForRes = conds[i / layersPerResolution]
      h = layer.noiseToSample(noise: h, cond: condForRes)
      if (i + 1) % layersPerResolution == 0 && i + 1 < layers.children.count {
        let noise = Tensor(randnLike: h) * temperature
        h = unsplitAlongTime(h, noise)
      }
    }
    return h
  }

  func sampleToNoise(sample: Tensor, cond: Tensor) -> (noises: [Tensor], logScale: Tensor) {
    let conds = computeConds(cond: cond)
    var h = sample
    var noises = [Tensor]()
    var logDet = Tensor(zeros: [h.shape[0]])
    for (i, layer) in layers.children.reversed().enumerated() {
      let condForRes = conds[i / layersPerResolution]

      #alwaysAssert(condForRes.shape[2] == h.shape[2])

      let (out, scale) = layer.sampleToNoise(sample: h, cond: condForRes)
      h = out
      logDet = logDet + scale

      if (i + 1) % layersPerResolution == 0 && i + 1 < layers.children.count {
        let (newH, noise) = splitAlongTime(h)
        noises.append(noise)
        h = newH
      }
    }
    return (noises: noises + [h], logScale: logDet)
  }

  func negativeLogLikelihood(sample: Tensor, cond: Tensor, quantization: Float = 1.0 / 255.0)
    -> Tensor
  {
    let (noises, logDet) = sampleToNoise(
      sample: sample + (Tensor(randLike: sample) - 0.5) * quantization, cond: cond)
    let bias = log(2.0 * Double.pi)
    var noiseProb = Tensor(zeros: [sample.shape[0]])
    for noise in noises {
      noiseProb = noiseProb - 0.5 * (bias + noise.pow(2)).flatten(startAxis: 1).sum(axis: 1)
    }
    let quantCorrection = log(quantization) * Float(sample.shape[2])
    return -(noiseProb + quantCorrection - logDet)
  }

  func computeConds(cond: Tensor) -> [Tensor] {
    var condH = cond
    var conds = [cond]
    for layer in condDownsample.children {
      condH = layer(condH)
      conds.append(condH)
    }
    return conds
  }
}

func splitAlongTime(_ x: Tensor) -> (Tensor, Tensor) {
  let split = x.reshape(x.shape[..<2] + [x.shape.last! / 2, 2]).chunk(axis: -1, count: 2).map {
    $0.squeeze(axis: -1)
  }
  return (split[0], split[1])
}

func unsplitAlongTime(_ x: Tensor, _ y: Tensor) -> Tensor {
  return Tensor(stack: [x, y], axis: -1).flatten(startAxis: -2)
}
