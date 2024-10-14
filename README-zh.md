<h1 align="center">MNP-SVC</h1>
<p align="center">
最小化噪声相位谐波源歌声转换系统 [版本2]
</p>

---

Language: [English](./README.md) \| [**简体中文**](#) \| [한국어\*](./README-ko.md) \| [日本語](./README-ja.md)

(\*: "机器翻译。欢迎提交母语者翻译的 PR！)

## 📒 简介

MNP-SVC 是一个开源的歌声转换项目，旨在开发可在个人电脑上广泛使用的免费 AI 变声软件。此项目是基于[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)仓库。

与原版相比，它并不会依赖外部声码器或扩散模型，而是通过(DP)WavLM 提高了降噪能力，改进了对高音的处理，并改善了输出结果。

此外，还有许多改进和实现。
（例如，loss的变更、快速插值方法、并且减少了说话特征泄漏的预训练方法等）
（例如，简单的语调调整等）

这个仓库主要关注以下改进：

- 使单个模型能够同时学习多个说话人
  - 抑制发声源的说话特征，使其适应目标说话人的特征
  - 同时保持较小容量的模型大小
- 获得更自然、更流畅的输出
  - 同时保持占用计算成本不过高

MNP 代表：Minimized-and-Noised-Phase harmonic source。

经过一些实验，我基于经验假设，不自然和有点刺耳的转换结果可能是由于相位是线性的，所以将声音合成器的谐波源从线性相位 sinc 函数改为应用窗函数的最小相位 sinc 函数。

（有可能线性相位使滤波器的学习变得困难。）

但这是合理的，因为所有自然产生的声音，包括人声，都是最小相位的。

我还改进了声学模型：Noised-Phase Harmonic Source
（我不是学专业的，但是我这样命名它。）

（目前，Noised-Phase 元素尚未使用。）

与 DDSP-SVC 的 AI 模型结构的主要区别是：

- 将卷积层（包括 ResNet）改为参考 ConvNeXt V2，并引入类似 GLU 的结构
- 使用说话人嵌入（也可以禁用。在这种情况下，似乎会降低对目标说话人特征的拟合性能。）
  - 使用嵌入时，如果用多说话人数据集训练，可能可以实现少样本转换。
- 在合并 F0、相位和说话人嵌入后添加卷积层。

免责声明：请确保 MNP-SVC 模型仅由**合法获得的正规数据**训练，不要将其用于非法目的合成语音。
本仓库的作者对使用这些模型检查点或输出音频所造成的侵权、违规、欺诈等非法行为不承担任何责任。

## 1. 🔨 构建依赖环境

### （针对 Windows 用户）简易设置

双击`dl-models.bat`和`launch.bat`。这个脚本在首次运行时会：

1. 下载预训练模型（dl-models.bat）
2. 下载[MicroMamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
3. 解压下载的文件并构建便携式 Python 环境
4. 为该 Python 环境安装依赖包

之后，您可以使用运行此脚本启动的控制台。

### （针对其他操作系统用户）手动设置

#### 1-1. 使用 pip 安装依赖

首先准备一个可以运行 Python 的环境（Windows 用户可以使用[WinPython](https://winpython.github.io/)等。
作者也在使用这个，并在 venv 中创建虚拟环境进行开发。），然后按照[PyTorch 官方网站](https://pytorch.org/)的步骤安装适合您环境的 PyTorch。

之后，运行：

```bash
pip install -r requirements/main.txt
```

来安装依赖包。

作者仅在 python 3.11.9/3.12.2（windows）+ cuda 12.1 + torch 2.4.1 环境下确认过运行。太旧或太新的版本可能无法运行。

#### 1-2. 下载预训练模型

- 特征编码器：

  1. 下载预训练的 [DPWavLM](https://huggingface.co/pyf98/DPHuBERT/blob/main/DPWavLM-sp0.75.pth) 编码器，并将其放在 `models/pretrained/dphubert` 文件夹下。

  2. 下载预训练的 [wespeaker-voxceleb-resnet34-LM (pyannote.audio 移植版)](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/) 说话人嵌入提取器（包括 [pytorch_model.bin](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/pytorch_model.bin) 和 [config.yaml](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/config.yaml)），并将其放在 `models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM` 文件夹下。

     - 或者打开配置文件（configs/combsub-mnp.yaml 或您想使用的其他配置文件），将 `data.spk_embed_encoder_ckpt` 的值更改为 `pyannote/wespeaker-voxceleb-resnet34-LM`。这样可以自动从 Hugging Face 模型库下载。

- 音高提取器：

  1. 下载预训练的 [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) 提取器，并将其解压到 `models/pretrained/` 文件夹中。

- MNP-SVC 预训练模型：

  下载 [预训练权重](https://huggingface.co/TylorShine/MNP-SVC-v2-pretrained/tree/main)，并将其放在 `models/pretrained/mnp-svc/` 文件夹下。

  - 也可以使用 [仅预训练少量卷积层的模型](https://github.com/TylorShine/MNP-SVC/releases/download/v0.0.1/model_0.pt)。这个模型没有训练声音特征、说话人分布等。

## 2. 🛠️ 预处理

将所有数据集 (音频文件) 放在 `dataset/audio`文件夹下。

支持同时学习多个说话人。如果您想训练多说话人模型，请将放置音频文件的文件夹名称设置为用下划线\_分隔的、作为说话人 ID 使用的 1 或更大的整数，和易于理解的名称。

文件夹结构示例如下:

```bash
# 第一位说话人
dataset/audio/1_first-speaker/aaa.wav
dataset/audio/1_first-speaker/bbb.wav
...
# 第二位说话人
dataset/audio/2_second-speaker/aaa.wav
dataset/audio/2_second-speaker/bbb.wav
...
```

对于**单一说话人** 模型，结构如下:

```bash
# 单一说话人数据集
dataset/audio/aaa.wav
dataset/audio/bbb.wav
...
```

然后，执行:

```bash
python sortup.py -c configs/combsub-mnp-san.yaml
```

这将自动将数据集分为 "train" 和 "test" 。如果您想调整此过程的参数，请运行 `python sortup.py -h` 查看帮助。
然后执行:

```bash
python preprocess.py -c configs/combsub-mnp-san.yaml
```

## 3. 🎓️ 训练

执行:

```bash
python train.py -c configs/combsub-mnp-san.yaml
```

您可以中途停止训练。只要不改变文件夹结构，您就可以再次运行相同的命令继续训练。

模型的微调也可以以相同的方式进行。

停止训练一次，重新预处理新的数据集，调整批量大小或学习率等，然后运行相同的命令。

## 4. 📉 可視化

```bash
# 使用 tensorboard 查看训练进度
tensorboard --logdir=dataset/exp
```

输出样本在首次测试后可见。

## 5. 🗃️ 非实时 VC

```bash
python main.py -i <输入wav文件> -m <model_num.pt> -o <输出wav文件> -k <音高偏移> -into <语调曲线> -id <说话人ID>
```

音高偏移: 以半音为单位。
语调曲线: 1.0 表示与输入相同的音高(默认)，减小会使音高更平坦(更平静)，增大会使音高更动态(更情感化)。

有关音高提取和响应阈值等其他选项，请运行:

```bash
python main.py -h
```

## 6. 🎤 实时 VC

运行以下命令启动简单的 GUI:

```bash
python gui.py
```

前端使用移动窗口、交叉淡入淡出和基于 SOLA 的切换等技术，在保持低延迟和低负载的同时，尽可能接近非实时的质量。

## 7. 📦️ 导出到 ONNX

使用以下命令导出为 ONNX 格式:

```bash
python -m tools.export_onnx -i <model_num.pt>
```

输出将在与输入文件相同的目录中，命名为 `model_num.onnx`。
有关其他选项，请运行 `python -m tools.export_onnx -h` 查看。 导出的 onnx 文件可以在实时 VC 和非实时 VC 中以相同的方式使用。目前仅支持 CPU 推理。

## 8. ⚖️ 许可证

[MIT License](LICENSE)

## 9. ✅️TODOs

- [x] 添加导出到 ONNX 格式的代码
- [ ] 创建 WebUI

## 10. 🙏 鸣谢

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)

  - 许多代码和想法来自这个仓库。非常感谢！Thanks a lot!

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)

- [DPHuBERT](https://github.com/pyf98/DPHuBERT)

- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)

- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

- [BigVSAN](https://github.com/sony/bigvsan)