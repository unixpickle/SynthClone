import Foundation
import HCBacktrace
import Honeycrisp

class FlowLayer: Trainable {
  let isEven: Bool

  @Child var condConv: Conv1D
  @Child var conv1: Conv1D
  @Child var conv2: Conv1D

  init(isEven: Bool, condChannels: Int, hiddenChannels: Int) {
    self.isEven = isEven
    super.init()
    condConv = Conv1D(
      inChannels: condChannels, outChannels: hiddenChannels, kernelSize: 3, stride: 2,
      padding: .allSides(1))
    conv1 = Conv1D(
      inChannels: 1, outChannels: hiddenChannels, kernelSize: 3, stride: 1, padding: .same)
    conv2 = Conv1D(
      inChannels: hiddenChannels, outChannels: 2, kernelSize: 3, stride: 1, padding: .same)
    for (_, var p) in conv2.parameters {
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

    let split = x.reshape(x.shape[..<2] + [x.shape.last! / 2, 2]).chunk(axis: -1, count: 2).map {
      $0.squeeze(axis: -1)
    }
    let (inputs, toModify) =
      if isEven {
        (split[0], split[1])
      } else {
        (split[1], split[0])
      }
    let scaleAndBias = conv2((conv1(inputs) + condConv(cond)).gelu()).chunk(axis: 1, count: 2)
    let (scale, bias) = (scaleAndBias[0], scaleAndBias[1])
    let outputs =
      if invert {
        (toModify - bias) / scale.exp()
      } else {
        toModify.mul(scale.exp(), thenAdd: bias)
      }
    let combinedOut =
      if isEven {
        Tensor(stack: [inputs, outputs], axis: -1).flatten(startAxis: -2)
      } else {
        Tensor(stack: [outputs, inputs], axis: -1).flatten(startAxis: -2)
      }
    return (out: combinedOut, logScale: scale.sum(axis: -1).squeeze(axis: 1))
  }
}

class FlowModel: Trainable {
  @Child var layers: TrainableArray<FlowLayer>

  init(condChannels: Int, layerCount: Int = 6, hiddenChannels: Int = 64) {
    super.init()
    layers = TrainableArray(
      (0..<layerCount).map { i in
        FlowLayer(isEven: i % 2 == 0, condChannels: condChannels, hiddenChannels: hiddenChannels)
      })
  }

  func noiseToSample(noise: Tensor? = nil, cond: Tensor) -> Tensor {
    var h = noise ?? Tensor(randn: [cond.shape[0], 1, cond.shape[2]])
    for layer in layers.children {
      h = layer.noiseToSample(noise: h, cond: cond)
    }
    return h
  }

  func sampleToNoise(sample: Tensor, cond: Tensor) -> (out: Tensor, logScale: Tensor) {
    var h = sample
    var logDet = Tensor(zeros: [h.shape[0]])
    for layer in layers.children.reversed() {
      let (out, scale) = layer.sampleToNoise(sample: h, cond: cond)
      h = out
      logDet = logDet + scale
    }
    return (out: h, logScale: logDet)
  }

  func negativeLogLikelihood(sample: Tensor, cond: Tensor, quantization: Float = 1.0 / 255.0)
    -> Tensor
  {
    let (noise, logDet) = sampleToNoise(
      sample: sample + (Tensor(randLike: sample) - 0.5) * quantization, cond: cond)
    let bias = log(2.0 * Double.pi)
    let noiseProb = -0.5 * (bias + noise.pow(2)).sum(axis: 2).squeeze(axis: 1)
    let quantCorrection = log(quantization) * Float(sample.shape[2])
    return -(noiseProb + quantCorrection - logDet)
  }
}
