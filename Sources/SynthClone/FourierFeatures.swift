import Foundation
import HCBacktrace
import Honeycrisp

class FourierFeatures {
  let bucketCount: Int
  let freqs: Tensor
  let biases: Tensor

  init(bucketCount: Int = 32, minCoeff: Float = 1.0, maxCoeff: Float = 100.0) {
    self.bucketCount = bucketCount
    #alwaysAssert(bucketCount % 2 == 0)

    let logMinCoeff = log(minCoeff)
    let logMaxCoeff = log(maxCoeff)

    freqs =
      ((Tensor(data: 0..<(bucketCount / 2), dtype: .float32) / Float(bucketCount / 2 - 1))
      * (logMaxCoeff - logMinCoeff) + logMinCoeff)
      .exp()
      .reshape([bucketCount / 2, 1])
    biases = Tensor(data: [0.0, Double.pi / 2], dtype: .float32)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = x.unsqueeze(axis: 2)
    h = h.mul(freqs.unsqueeze(axis: -1), thenAdd: biases.unsqueeze(axis: -1))
    h = h.sin()
    h = h.flatten(startAxis: 1, endAxis: 2)
    return h
  }
}
