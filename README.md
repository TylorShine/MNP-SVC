<h1 align="center">MNP-SVC</h1>
<p align="center">
Minimized-and-Noised-Phase harmonic source Singing Voice Convertion
</p>

---

Language: [**English**](./README.md) | [简体中文* (comming soon)](#) | [한국어* (comming soon)](#) | [日本語](./README-ja.md)  

(*: machine translation. PR is welcome to native translations!)


## 📒Introduction

MNP-SVC is a open source singing voice conversion project dedicated to the development of free AI voice changer software that can be popularized on personal computers. These aims are inherited from an original repository. ([DDSP-SVC](https://github.com/yxlllc/DDSP-SVC))

Compared with an original, not to use the external vocoder and diffusion models, improved noise robustness thank by (DP)WavLM and change unvoiced pitch handling, and improved result (my feeling, subjectively). And there are many improvements (e.g. change losses, fast interpolation method, pretraining method for decrease original speaker feature leakage), implementations (e.g. easy intonation curve tweak).

This repo focus to improvement:
- learning multiple speakers at once into a single model
  - reduce an original speaker's features and fit to target speaker's one
  - still keep even small model size
- more natural and smooth output result
  - and computational cost still keep not heavily

MNP refers: Minimized-and-Noised-Phase harmonic source.  
After some experimentation, I changed harmonic source signal of synthesizer from linear-phase sinc to minimized-phase windowed sinc because I put the assumption that the unnatural and slightly not catchy sensations of the result may be due to the fact that the phase is linear. (And maybe, that thing made learning filters harder.)
This is appropriate that I think because all naturally occurring sounds, including human voices, are in minimum phase.  
And improved acoustic model: The Noised-Phase Harmonic Source (named by me, I'm not a scholar.).


Different of model structure from DDSP-SVC is about:
- Use the ConvNeXt-V2-like convolution layers
- Use speaker embedding (optionally you can disable it)
- Use conv layer after combining F0, phase and speaker embed


Disclaimer: Please make sure to only train DDSP-SVC models with **legally obtained authorized data**, and do not use these models and any audio they synthesize for illegal purposes. The author of this repository is not responsible for any infringement, fraud and other illegal acts caused by the use of these model checkpoints and audio.


## 1. 🔨Installing the dependencies

We recommend first installing PyTorch from the [official website](https://pytorch.org/), then run:

```bash
pip install -r requirements/main.txt
```

NOTE : I only test the code using python 3.11.8/3.12.2 (windows) + cuda 11.8 + torch 2.2.1, too new or too old dependencies may not work.


## 2. 📝Configuring your model

### 2-1. Download pre-trained models
- Feature Encoders:

  1. Download the pre-trained [DPWavLM](https://huggingface.co/pyf98/DPHuBERT/blob/main/DPWavLM-sp0.75.pth) encoder and put it under `models/pretrained/dphubert` folder.
  1. Download the pre-trained [wespeaker-voxceleb-resnet34-LM (pyannote.audio ported)](
https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/) speaker embed extractor (both [pytorch_model.bin](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/pytorch_model.bin) and [config.yaml](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/config.yaml)) and puts it under `models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM` folder.
      - or open configs (configs/combsub-mnp.yaml or you wanna use), and change `data.spk_embed_encoder_ckpt` value to `pyannote/wespeaker-voxceleb-resnet34-LM`. this allows download from huggingface model hub's one automatically.

- Pitch extractor:

  1. Download the pre-trained [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) extractor and unzip it into `models/pretrained/` folder.

- MNP-SVC pre-trained model:

  Download the (only feature conv) pre-trained [model](https://github.com/TylorShine/MNP-SVC/releases/download/v0.0.0/model_0.pt). Use this later, keep in mind.


## 3. 🛠️Preprocessing

Put all the dataset (audio clips) in the below directory: `dataset/audio`.


NOTE: Multi-speaker training is supported. If you want to train a **multi-speaker** model, audio folders need to be named with **positive integers** to represent speaker ids and friendly name separated with a underscore "_", the directory structure is like below:

```bash
# the 1st speaker
dataset/audio/1_first-speaker/aaa.wav
dataset/audio/1_first-speaker/bbb.wav
...
# the 2nd speaker
dataset/audio/2_second-speaker/aaa.wav
dataset/audio/2_second-speaker/bbb.wav
...
```

The directory structure of the **single speaker** model is also supported, which is like below:

```bash
# single speaker dataset
dataset/audio/aaa.wav
dataset/audio/bbb.wav
...
```


then run

```bash
python sortup.py -c configs/combsub-mnp.yaml
```

to divide your datasets to "train" and "test" automatically. If you wanna adjust some parameters, run `python sortup.py -h` to help you.
After that, then run

```bash
python preprocess.py -c configs/combsub-mnp.yaml
```

After done, put a MNP-SVC pre-trained model (`model_0.pt`) to under `dataset/exp/combsub-mnp/`.


## 4. 🎓️Training

```bash
python train.py -c configs/combsub-mnp.yaml
```

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.


## 5. 📉Visualization

```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```

Test audio samples will be visible in TensorBoard after the first validation.


## 6. 🗃️Non-real-time VC

```bash
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange> -into <intonation curve> -id <speaker_id>
```

keychange: semitones  
intonation curve: 1.0 means follow original pitch (default), more small to flat (calm), more large to dynamic (excite)  

Other options about the f0 extractor and response threhold，see:

```bash
python main.py -h
```


## 7. 🎤Real-time VC

Start a simple GUI with the following command:

```bash
python gui.py
```

The front-end uses technologies such as sliding window, cross-fading, SOLA-based splicing and contextual semantic reference, which can achieve sound quality close to non-real-time synthesis with low latency and resource occupation.


## 8. 🙏Acknowledgement

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
  - Many codes and ideas based on, thanks a lot!

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)

- [DPHuBERT](https://github.com/pyf98/DPHuBERT)

- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)
  
  