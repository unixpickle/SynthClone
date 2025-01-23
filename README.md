# SynthClone

Let's train a model to clone the Mac's built-in speech synthesizer.

This is largely implemented based on [HCText2Image](https://github.com/unixpickle/HCText2Image), but simplified a lot.

# Usage

## Using existing models

To run the existing model, download these checkpoints

 * [vqvae_jan03.plist](https://data.aqnichol.com/SynthClone/vqvae_jan03.plist) - the VQ encoder and decoder
 * [xf_jan15.plist](https://data.aqnichol.com/SynthClone/xf_jan15.plist) - the final transformer model

You can then run the command

```
swift run -c release SynthClone server vqvae_jan03.plist xf_jan15.plist
```

to launch a server on your Mac listening on `http://localhost:8080`. This will provide a simple web interface to convert text to speech.

## Training a transformer

To train a transformer with a pretokenized dataset, you can run a command like:

```
swift run -c release SynthClone transformer <data_dir> <vqvae_path.plist> output_path.plist
```

You can download a pre-trained VQ-VAE, or use one you've trained from scratch. If you use the pre-trained VQ-VAE, you can download the pretokenized data [here](https://data.aqnichol.com/SynthClone/tokenized.tar).
