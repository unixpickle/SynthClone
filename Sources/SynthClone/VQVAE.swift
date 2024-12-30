import HCBacktrace
import Honeycrisp

class DownsampleBlock: Trainable {
  @Child var norm: GroupNorm
  @Child var conv1: Conv1D
  @Child var conv2: Conv1D
  @Child var skip: Conv1D

  init(inChannels: Int, outChannels: Int) {
    super.init()
    self.norm = GroupNorm(groupCount: 32, channelCount: inChannels)
    self.conv1 = Conv1D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: 3, padding: .same)
    self.conv2 = Conv1D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: 3, stride: 2,
      padding: .allSides(1))
    self.skip = Conv1D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: 1, stride: 2)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = norm(x)
    h = conv1(h)
    h = h.gelu()
    h = conv2(h)
    h = h + skip(x)
    return h
  }
}

class UpsampleBlock: Trainable {
  @Child var norm: GroupNorm
  @Child var conv1: Conv1D
  @Child var conv2: Conv1D
  @Child var skip: Conv1D

  init(inChannels: Int, outChannels: Int) {
    super.init()
    self.norm = GroupNorm(groupCount: 32, channelCount: inChannels)
    self.conv1 = Conv1D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: 3, padding: .same)
    self.conv2 = Conv1D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: 3, padding: .same)
    self.skip = Conv1D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: 1)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    // Neighbor upsampling for x.
    let x = x.unsqueeze(axis: -1).repeating(axis: -1, count: 2).flatten(startAxis: -2)

    var h = norm(x)
    h = conv1(h)
    h = h.gelu()
    h = conv2(h)
    h = h + skip(x)
    return h
  }
}

class VQEncoder: Trainable {
  var ff: FourierFeatures
  @Child var inProj: Conv1D
  @Child var blocks: TrainableArray<DownsampleBlock>
  @Child var outNorm: GroupNorm
  @Child var outProj: Conv1D

  init(inChannels: Int, outChannels: Int, downsamples: Int, maxChannels: Int = 256) {
    #alwaysAssert(inChannels == 1)
    ff = FourierFeatures()
    super.init()
    self.inProj = Conv1D(
      inChannels: ff.bucketCount, outChannels: 64, kernelSize: 3, padding: .same)

    var blockArr = [DownsampleBlock]()
    var ch = 64
    for _ in 0..<downsamples {
      blockArr.append(
        DownsampleBlock(inChannels: min(maxChannels, ch), outChannels: min(maxChannels, ch * 2)))
      ch *= 2
    }
    self.blocks = TrainableArray(blockArr)
    self.outNorm = GroupNorm(groupCount: 32, channelCount: min(maxChannels, ch))
    self.outProj = Conv1D(
      inChannels: min(ch, maxChannels), outChannels: outChannels, kernelSize: 1)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = ff(h)
    h = inProj(h)
    for layer in blocks.children {
      h = layer(h)
    }
    h = outNorm(h)
    h = outProj(h)
    return h
  }
}

class VQDecoder: Trainable {
  @Child var inProj: Conv1D
  @Child var blocks: TrainableArray<UpsampleBlock>
  @Child var outNorm: GroupNorm
  @Child var outProj: Conv1D

  init(inChannels: Int, outChannels: Int, upsamples: Int, maxChannels: Int = 256) {
    super.init()
    self.inProj = Conv1D(
      inChannels: inChannels, outChannels: min(maxChannels, 64 * (1 << upsamples)),
      kernelSize: 3, padding: .same)

    var blockArr = [UpsampleBlock]()
    var ch = 64 * (1 << upsamples)
    for _ in 0..<upsamples {
      blockArr.append(
        UpsampleBlock(inChannels: min(ch, maxChannels), outChannels: min(ch / 2, maxChannels)))
      ch /= 2
    }
    self.blocks = TrainableArray(blockArr)
    self.outNorm = GroupNorm(groupCount: 32, channelCount: min(maxChannels, ch))
    self.outProj = Conv1D(
      inChannels: min(maxChannels, ch), outChannels: outChannels, kernelSize: 3,
      padding: .same)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = self.inProj(x)
    for layer in blocks.children {
      h = layer(h)
    }
    h = outNorm(h)
    h = h.gelu()
    h = self.outProj(h)
    return h
  }
}

class VQBottleneck: Trainable {
  struct Losses {
    let commitmentLoss: Tensor
    let codebookLoss: Tensor
  }

  struct Output {
    let straightThrough: Tensor
    let codes: Tensor
    let losses: Losses
  }

  let vocab: Int
  let channels: Int

  @Buf var usageCounter: Tensor
  @Param var dictionary: Tensor

  init(vocab: Int, channels: Int) {
    self.vocab = vocab
    self.channels = channels
    super.init()
    self.usageCounter = Tensor(zeros: [vocab], dtype: .int64)
    self.dictionary = normalize(Tensor(randn: [vocab, channels]))
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Output {
    let dictionary = normalize(self.dictionary)

    let batch = x.shape[0]
    let channels = x.shape[1]
    let spatialShape = Array(x.shape[2...])

    let vecs = normalize(x.move(axis: 1, to: -1).flatten(endAxis: -2))
    let codes = nearestIndices(vecs, dictionary)

    if mode == .training {
      usageCounter =
        usageCounter + Tensor(onesLike: codes).scatter(axis: 0, count: vocab, indices: codes)
    }

    let selection = dictionary.gather(axis: 0, indices: codes)
    let out = selection.reshape([batch] + spatialShape + [channels]).move(axis: -1, to: 1)
    return Output(
      straightThrough: out.noGrad() + (x - x.noGrad()),
      codes: codes.reshape([batch] + spatialShape),
      losses: Losses(
        commitmentLoss: (out.noGrad() - x).pow(2).mean(),
        codebookLoss: (out - x.noGrad()).pow(2).mean()
      )
    )
  }

  func embed(_ x: Tensor) -> Tensor {
    return normalize(dictionary).gather(axis: 0, indices: x.flatten()).reshape(x.shape + [channels])
  }

  func revive(_ x: Tensor) -> Tensor {
    Tensor.withGrad(enabled: false) {
      // Make sure there's enough centers.
      var x = x
      while x.shape[0] < vocab {
        x = Tensor(concat: [x, x + Tensor(randnLike: x) * 0.001])
      }

      let shuffleInds = Tensor(data: Array((0..<x.shape[0]).shuffled()[..<vocab]), shape: [vocab])
      let newCenters = x.gather(axis: 0, indices: shuffleInds)

      let mask = (usageCounter > 0).unsqueeze(axis: 1).expand(as: dictionary)
      dictionary = mask.when(isTrue: dictionary, isFalse: normalize(newCenters))

      let reviveCount = (usageCounter == 0).cast(.int64).sum()
      usageCounter = Tensor(zerosLike: usageCounter)
      return reviveCount
    }
  }
}

func normalize(_ x: Tensor) -> Tensor {
  x / x.pow(2).sum(axis: 1, keepdims: true).clamp(min: 1e-5).sqrt()
}

func nearestIndices(_ vecs: Tensor, _ centers: Tensor) -> Tensor {
  Tensor.withGrad(enabled: false) {
    let dots = Tensor.matmul(a: vecs, transA: false, b: centers, transB: true, transOut: false)
    let vecsNorm = vecs.pow(2).sum(axis: 1, keepdims: true).expand(as: dots)
    let dictNorm = centers.pow(2).sum(axis: 1).unsqueeze(axis: 0).expand(as: dots)
    let dists = vecsNorm + dictNorm - 2 * dots
    return dists.argmin(axis: 1)
  }
}

class VQVAE: Trainable {
  @Child var encoder: VQEncoder
  @Child var bottleneck: VQBottleneck
  @Child var decoder: VQDecoder
  @Child var outFlow: FlowModel

  init(channels: Int, vocab: Int, latentChannels: Int, downsamples: Int, condChannels: Int = 4) {
    super.init()
    self.encoder = VQEncoder(
      inChannels: channels, outChannels: latentChannels, downsamples: downsamples)
    self.bottleneck = VQBottleneck(vocab: vocab, channels: latentChannels)
    self.decoder = VQDecoder(
      inChannels: latentChannels, outChannels: condChannels, upsamples: downsamples)
    self.outFlow = FlowModel(condChannels: condChannels)
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> (
    nll: Tensor, vqLosses: VQBottleneck.Losses
  ) {
    var h = x
    h = encoder(h)
    let vqOut = bottleneck(h)
    h = vqOut.straightThrough
    h = decoder(h)
    let nll = outFlow.negativeLogLikelihood(sample: x, cond: h)
    return (nll: nll / x.shape[2], vqLosses: vqOut.losses)
  }

  func sampleReconstruction(_ x: Tensor, temperature: Float = 1.0) -> Tensor {
    var h = x
    h = encoder(h)
    let vqOut = bottleneck(h)
    h = vqOut.straightThrough
    h = decoder(h)
    return outFlow.noiseToSample(cond: h, temperature: temperature)
  }

  func features(_ x: Tensor) -> Tensor {
    encoder(x).move(axis: 1, to: -1).flatten(endAxis: -2)
  }
}
