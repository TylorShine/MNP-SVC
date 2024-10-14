<h1 align="center">MNP-SVC</h1>
<p align="center">
Minimized-and-Noised-Phase harmonic source Singing Voice Convertion [v2]
</p>

---

Language: [English](./README.md) \| [简体中文](./README-zh.md) \| [한국어\*](./README-ko.md) \| [**日本語**](#)  

(\*: 機械翻訳です。 ネイティブ翻訳のPRを歓迎します！)


## 📒はじめに

MNP-SVC は、自由なAIボイスチェンジャーソフトウェアをPCで実行可能にするオープンソースの歌唱音声変換プロジェクトです。これらの目的は多くのコードのベースとして使わせて頂いた[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)レポジトリから継承しています。 

オリジナルと比較して、外部ボコーダーや拡散モデルを使用せず、(DP)WavLMによる対ノイズ堅牢性の向上、ピッチのない音声のハンドリングの変更、そして出力結果の改善(主観です)をしています。  
さらに、多くの改善 (例えば、lossの変更、高速な補間メソッド、変換元話者の特徴漏れを減らすための事前学習メソッドなど)、実装 (例えば、簡単な抑揚の調整など) があります。

このレポジトリでは、以下の改善を主眼においています:
- 単一のモデルで複数話者を同時に学習できるようにする
  - 変換元話者の特徴を抑え、変換先話者の特徴にフィットさせる
  - それでも小さなモデルサイズをキープする
- もっと自然でスムースな出力を得る
  - それでいて計算コストは重すぎないよう維持する

MNPとは: Minimized-and-Noised-Phase harmonic source です。  
いくつかの実験の結果、不自然で少し耳につく変換結果はおそらく位相が線形であることに起因するであろうと(経験から)仮定して、サウンド合成器のハーモニックソースを線形位相sinc関数から窓関数を適用した最小位相sinc関数に変更しました。(そしておそらく、線形位相であることはフィルタの学習を難しくしています。)  
このことは、人の音声を含む自然で発生しうるすべての音は最小位相であることからも適切であると考えます。  
音響モデルも改善しました: Noised-Phase Harmonic Source (私は学者ではありませんが、そう名付けました。)
(今のところ、Noised-Phase要素は使われていません。)


DDSP-SVCとのAIモデルの構造の違いは、大まかに:
- 畳み込み層(ResNet含む)を ConvNeXt V2 を参考にし、GLUのような構造を取り入れた形に変更
- 話者埋め込みを使用 (無効にすることもできます。その場合は変換先話者への特徴のフィット性能が落ちるようです。)
  - 埋め込み使用時、多数話者のデータセットで学習すれば、few-shot変換が可能かもしれません。
- F0、位相、話者埋め込みをまとめたあとに畳み込み層を追加しています。


免責事項: MNP-SVCモデルが**合法的に入手した正規のデータ**のみによって学習されたことを確認の上、合法でない目的のために音声を合成するのに使わないでください。このリポジトリの作者はこれらのモデルチェックポイントや出力音声を使用したことによる侵害、違反、詐欺等の違法行為に責任を負わないものとします。


## 1. 🔨依存環境の構築

### (Windowsユーザー向け) 簡易セットアップ

`dl-models.bat` と `launch.bat` をダブルクリックします。このスクリプトは初回実行時に:

1. 事前学習モデルのダウンロード (dl-models.bat)
1. [MicroMamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) のダウンロード
1. ダウンロードしたファイルを展開しポータブルなPython環境を構築
1. そのPython環境に対し、依存パッケージをインストール


を実行します.  
次回からは、このスクリプトを実行して起動したコンソールを使用できます。


### (その他のOSのユーザー向け) 手動セットアップ

#### 1-1. pip で依存関係をインストール
事前にPythonを実行できる環境を用意し(Windowsの方は[WinPython](https://winpython.github.io/)など。作者もこちらでvenvで仮想環境を作成して開発しています。)、はじめに [PyTorch 公式ウェブサイト](https://pytorch.org/) の手順でお使いの環境にあったPyTorchをインストールしてください。  
その後:

```bash
pip install -r requirements/main.txt
```

を実行して依存パッケージをインストールします。

作者は python 3.11.9/3.12.2 (windows) + cuda 12.1 + torch 2.4.1 のみで実行を確認しています。古すぎたり新しすぎるバージョンでは動かないかもしれません。


#### 1-2. 事前学習モデルのダウンロード

事前学習モデルをダウンロードします。
- 特徴抽出器:

  1. [DPWavLM](https://huggingface.co/pyf98/DPHuBERT/blob/main/DPWavLM-sp0.75.pth) をダウンロードし、`models/pretrained/dphubert`フォルダ以下に配置します。
  1. [wespeaker-voxceleb-resnet34-LM (pyannote.audio版)](
https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/)  ([pytorch_model.bin](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/pytorch_model.bin) と [config.yaml](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/config.yaml)) をダウンロードし、`models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM`フォルダ以下に設置します。
      - もしくは、コンフィグファイル (configs/combsub-mnp.yaml もしくは使用したいもの)の`data.spk_embed_encoder_ckpt`の値を`pyannote/wespeaker-voxceleb-resnet34-LM`に変更してください。この変更をするとHuggingFace Model Hubのモデルを自動的にダウンロードしてキャッシュするようになります。

- ピッチ抽出器:

  1. [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) をダウンロードし、`models/pretrained/`以下にzipファイルを展開します。 (`models/pretrained/rmvpe/...`のようになります。)

- MNP-SVC 事前学習モデル:

  [事前学習モデル](https://huggingface.co/TylorShine/MNP-SVC-v2-pretrained/tree/main) をダウンロードし、`models/pretrained/mnp-svc/`以下にそのまま配置します。
  - 必要に応じて [一部の畳み込み層のみの事前学習モデル](https://github.com/TylorShine/MNP-SVC/releases/download/v0.0.1/model_0.pt) も使用可能です。こちらは声質や話者の特徴分布が学習されていないものです。


## 2. 🛠️事前処理

すべてのデータセット (音声ファイル) を`dataset/audio`フォルダ以下に配置します。 


複数話者の同時学習に対応しています。**複数話者**モデルを学習したい場合は、音声ファイルを配置するフォルダ名を **アンダーバー_で区切った、話者IDとして使用する1以上の整数と、わかりやすい名前** で構成してください。  
フォルダ構成例は以下のようになります:

```bash
# 1番目の話者
dataset/audio/1_first-speaker/aaa.wav
dataset/audio/1_first-speaker/bbb.wav
...
# 2番目の話者
dataset/audio/2_second-speaker/aaa.wav
dataset/audio/2_second-speaker/bbb.wav
...
```

**単一話者** モデルの場合は以下のようになります:

```bash
# 単一話者データセット
dataset/audio/aaa.wav
dataset/audio/bbb.wav
...
```


その後、

```bash
python sortup.py -c configs/combsub-mnp-san.yaml
```

を実行します。これは自動で "train" と "test" にデータセットを振り分けます。この際のパラメーターを調整したい場合は、 `python sortup.py -h` を実行してヘルプを確認してください。  
そして、

```bash
python preprocess.py -c configs/combsub-mnp-san.yaml
```

を実行します。

## 3. 🎓️学習

```bash
python train.py -c configs/combsub-mnp-san.yaml
```

を実行します。

途中で学習を止められます。フォルダ構成を変えない限り、もう一度同じコマンドで学習を継続できます。

モデルのファインチューニングも同じように行えます。一度学習を止めて、新しいデータセットをもう一度事前処理し、バッチサイズや学習率をチューニングするなどしてから同コマンドを実行してください。


## 4. 📉可視化

```bash
# 学習経過を tensorboard で確認する
tensorboard --logdir=dataset/exp
```

出力サンプルは初回のテスト後に見えます。


## 5. 🗃️非リアルタイムVC

```bash
python main.py -i <入力wavファイル> -m <model_num.pt> -o <出力先wavファイル> -k <キーシフト> -into <抑揚カーブ> -id <話者ID>
```

キーシフト: 半音単位です。  
抑揚カーブ: 1.0で入力と同じピッチ(デフォルトです)、小さくするとフラットに(落ち着いた感じに)、大きくするとダイナミックに(感情的に)なります。

ピッチ抽出や応答のしきい値など他のオプションは:

```bash
python main.py -h
```

で確認してください。


## 6. 🎤リアルタイムVC

以下のコマンドで簡易GUIを実行します:

```bash
python gui.py
```

フロントエンドは移動窓、クロスフェード、SOLAベースの切継ぎなどを使用し、低遅延と低負荷を維持しながら品質をノンリアルタイムと近づけるようにしています。


## 7. 📦️ONNXへのエクスポート

以下のコマンドでONNX形式にエクスポートします:

```bash
python -m tools.export_onnx -i <model_num.pt>
```

入力ファイルと同一ディレクトリに`model_num.onnx`のように出力されます。  
その他のオプションは `python -m tools.export_onnx -h` で確認してください。  
エクスポートされたonnxファイルは、リアルタイムVCや非リアルタイムVCで同じように使用できます。今のところCPU推論のみの対応です。


## 8. ⚖️ライセンス
[MIT License](LICENSE) です。


## 9. ✅️TODOs
- [x] ONNX 形式にエクスポートするコードの追加
- [ ] WebUI の作成


## 10. 🙏謝辞

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
  - 多くのコードとアイデアはこちらのレポジトリから来ています。ありがとうございます！ Thanks a lot!

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)

- [DPHuBERT](https://github.com/pyf98/DPHuBERT)

- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)

- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

- [BigVSAN](https://github.com/sony/bigvsan)
