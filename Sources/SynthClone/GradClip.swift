import Honeycrisp

public func clipGradients<T: NumericTensorElement>(model: Trainable, threshold: T) -> Tensor {
  var gradNorm = Tensor(data: [0.0])
  for (name, p) in model.parameters {
    if let g = p.grad {
      gradNorm = gradNorm + g.pow(2).sum()
    }
  }
  let n = gradNorm.sqrt()
  let scale = (threshold / n).clamp(max: 1)
  for (_, var p) in model.parameters {
    if let g = p.grad {
      p.grad = g * scale
    }
  }
  return n
}
