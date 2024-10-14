
<h1 align="center">MNP-SVC</h1>
<p align="center">
최소화 및 노이즈 위상 하모닉 소스 노래 음성 변환 [v2]
</p>

---

Language: [English](./README.md) | [简体中文](./README-zh.md) | [**한국어**\*](#) | [日本語](./README-ja.md)  

(\*: 기계 번역. 네이티브 번역에 대한 PR 환영합니다!)


## 📒소개

MNP-SVC는 개인용 컴퓨터에서 대중화될 수 있는 무료 AI 음성 변환 소프트웨어 개발에 전념하는 오픈 소스 노래 음성 변환 프로젝트입니다. 이러한 목표는 원래 저장소에서 계승되었습니다. ([DDSP-SVC](https://github.com/yxlllc/DDSP-SVC))

원본과 비교하여 외부 보코더와 확산 모델을 사용하지 않고, (DP)WavLM 덕분에 노이즈 강건성이 향상되었으며 무성음 피치 처리가 변경되었고, 결과가 개선되었습니다(제 느낌으로는 주관적으로). 또한 많은 개선 사항(예: 손실 변경, 빠른 보간 방법, 원래 화자 특징 누출을 줄이기 위한 사전 훈련 방법)과 구현(예: 쉬운 억양 곡선 조정)이 있습니다.

이 저장소는 다음 사항에 중점을 둡니다:
- 단일 모델에 여러 화자를 한 번에 학습
  - 원래 화자의 특징을 줄이고 목표 화자의 특징에 맞춤
  - 여전히 작은 모델 크기 유지
- 더 자연스럽고 부드러운 출력 결과
  - 그리고 계산 비용은 여전히 크지 않게 유지

MNP는 최소화 및 노이즈 위상 하모닉 소스를 의미합니다.  
몇 가지 실험 후, 합성기의 하모닉 소스 신호를 선형 위상 싱크에서 최소화 위상 윈도우 싱크로 변경했습니다. 이는 결과의 부자연스럽고 약간 잡히지 않는 감각이 위상이 선형이라는 사실 때문일 수 있다는 가정을 했기 때문입니다. (그리고 아마도 그것이 학습 필터를 더 어렵게 만들었을 수 있습니다.)
이는 인간의 목소리를 포함한 모든 자연적으로 발생하는 소리가 최소 위상에 있다고 생각하기 때문에 적절합니다.  
그리고 개선된 음향 모델: 노이즈 위상 하모닉 소스 (제가 명명한 것입니다. 저는 학자가 아닙니다.).
(현재 노이즈 위상 기능은 사용되지 않습니다.)


DDSP-SVC와의 모델 구조 차이점:
- GLU와 유사한 구조를 가진 ConvNeXt-V2와 유사한 컨볼루션 레이어 사용
- 화자 임베딩 사용 (선택적으로 비활성화 가능)
- F0, 위상 및 화자 임베드를 결합한 후 컨볼루션 레이어 사용


면책 조항: DDSP-SVC 모델을 **합법적으로 획득한 승인된 데이터**로만 훈련시키고, 이러한 모델과 그들이 합성한 오디오를 불법적인 목적으로 사용하지 않도록 주의하십시오. 이 저장소의 작성자는 이러한 모델 체크포인트와 오디오의 사용으로 인한 침해, 사기 및 기타 불법 행위에 대해 책임을 지지 않습니다.


## 1. 🔨의존성 설치

### (Windows 사용자를 위한) 간편 설정

`dl-models.bat`와 `launch.bat`를 더블 클릭하기만 하면 됩니다. 이 스크립트는 다음을 수행합니다:
1. 사전 훈련된 모델이 없는 경우 다운로드 (dl-models.bat)
1. [MicroMamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) 다운로드
1. 다운로드한 아카이브 추출 및 휴대용 Python 환경 생성
1. 휴대용 Python 환경에 필요한 패키지 설치

첫 실행 시 이 작업을 수행합니다.  
다음에는 이 스크립트를 실행하고 이 콘솔을 사용할 수 있습니다.


### (다른 OS 사용자를 위한) 수동 설정

#### 1-1. pip로 의존성 설치
먼저 [공식 웹사이트](https://pytorch.org/)에서 PyTorch를 설치하는 것을 권장합니다. 그런 다음 실행:

```sh
pip install -r requirements/main.txt
```


참고: 저는 python 3.11.9/3.12.2 (windows) + cuda 12.1 + torch 2.4.1만 사용하여 코드를 테스트했습니다. 너무 새롭거나 오래된 의존성은 작동하지 않을 수 있습니다.


#### 1-2. 사전 훈련된 모델 다운로드
- 특징 인코더:

  1. 사전 훈련된 [DPWavLM](https://huggingface.co/pyf98/DPHuBERT/blob/main/DPWavLM-sp0.75.pth) 인코더를 다운로드하여 `models/pretrained/dphubert` 폴더에 넣습니다.
  1. 사전 훈련된 [wespeaker-voxceleb-resnet34-LM (pyannote.audio 포팅)](
https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/) 화자 임베드 추출기 ([pytorch_model.bin](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/pytorch_model.bin)과 [config.yaml](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/blob/main/config.yaml) 모두)를 다운로드하여 `models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM` 폴더에 넣습니다.
      - 또는 configs (configs/combsub-mnp.yaml 또는 사용하고자 하는 것)를 열고 `data.spk_embed_encoder_ckpt` 값을 `pyannote/wespeaker-voxceleb-resnet34-LM`으로 변경합니다. 이렇게 하면 huggingface 모델 허브에서 자동으로 다운로드할 수 있습니다.

- 피치 추출기:

  1. 사전 훈련된 [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) 추출기를 다운로드하여 `models/pretrained/` 폴더에 압축을 풉니다.

- MNP-SVC 사전 훈련 모델:

  [사전 훈련된 가중치](https://huggingface.co/TylorShine/MNP-SVC-v2-pretrained/tree/main)를 다운로드하여 `models/pretrained/mnp-svc/` 폴더에 넣습니다.
  - [몇 개의 컨볼루션 레이어만 사전 훈련된 모델](https://github.com/TylorShine/MNP-SVC/releases/download/v0.0.1/model_0.pt)도 사용 가능합니다. 이 모델은 음성 특성, 화자 분포 등을 훈련하지 않았습니다.


## 2. 🛠️전처리

모든 데이터셋(오디오 클립)을 다음 디렉토리에 넣습니다: `dataset/audio`.


참고: 다중 화자 훈련이 지원됩니다. **다중 화자** 모델을 훈련하려면 오디오 폴더의 이름을 화자 ID를 나타내는 **양의 정수**와 언더스코어 "_"로 구분된 친숙한 이름으로 지정해야 합니다. 디렉토리 구조는 다음과 같습니다:


```sh
# 첫 번째 화자
dataset/audio/1_first-speaker/aaa.wav
dataset/audio/1_first-speaker/bbb.wav
...
# 두 번째 화자
dataset/audio/2_second-speaker/aaa.wav
dataset/audio/2_second-speaker/bbb.wav
...
```

**단일 화자** 모델의 디렉토리 구조도 지원되며, 다음과 같습니다:

```sh
# 단일 화자 데이터셋
dataset/audio/aaa.wav
dataset/audio/bbb.wav
...
```


그런 다음 실행:

```sh
python sortup.py -c configs/combsub-mnp-san.yaml
```

데이터셋을 자동으로 "train"과 "test"로 나눕니다. 일부 매개변수를 조정하려면 `python sortup.py -h`를 실행하여 도움말을 확인하세요.
그 후, 다음을 실행합니다:

```sh
python preprocess.py -c configs/combsub-mnp-san.yaml
```


## 3. 🎓️훈련

```sh
python train.py -c configs/combsub-mnp-san.yaml
```

훈련을 안전하게 중단할 수 있으며, 동일한 명령줄을 실행하면 훈련이 재개됩니다.

또한 훈련을 먼저 중단한 다음 새 데이터셋을 다시 전처리하거나 훈련 매개변수(배치 크기, 학습률 등)를 변경한 후 동일한 명령줄을 실행하여 모델을 미세 조정할 수 있습니다.


## 4. 📉시각화

```sh
# tensorboard를 사용하여 훈련 상태 확인
tensorboard --logdir=exp
```

첫 번째 검증 후 TensorBoard에서 테스트 오디오 샘플을 볼 수 있습니다.

## 5. 🗃️비실시간 VC

```sh
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange> -into <intonation curve> -id <speaker_id>
```

keychange: 반음  
intonation curve: 1.0은 원래 피치를 따름(기본값), 더 작으면 평탄(차분), 더 크면 동적(흥분)  

f0 추출기 및 응답 임계값에 대한 기타 옵션은 다음을 참조하세요:

```sh
python main.py -h
```


## 6. 🎤실시간 VC

다음 명령으로 간단한 GUI를 시작합니다:

```sh
python gui.py
```


프론트엔드는 슬라이딩 윈도우, 크로스페이딩, SOLA 기반 스플라이싱 및 컨텍스트 의미 참조와 같은 기술을 사용하여 낮은 지연 시간과 리소스 점유로 비실시간 합성에 가까운 음질을 달성할 수 있습니다.


## 7. 📦️ONNX로 내보내기

다음 명령을 실행합니다:

```sh
python -m tools.export_onnx -i <model_num.pt>
```


입력 파일과 동일한 디렉토리에 `model_num.onnx`와 같은 이름으로 내보냅니다.  
기타 옵션은 `python -m tools.export_onnx -h`에서 확인할 수 있습니다.  
내보낸 onnx 파일은 실시간 및 비실시간 VC에 동일한 방식으로 사용할 수 있습니다. 현재는 CPU 추론만 지원됩니다.


## 8. ⚖️라이선스
[MIT License](LICENSE)


## 9. ✅️할 일
- [x] ONNX로 내보내기
- [ ] WebUI 만들기


## 10. 🙏감사의 말

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
  - 많은 코드와 아이디어가 기반이 되었습니다. 정말 감사합니다!

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)

- [DPHuBERT](https://github.com/pyf98/DPHuBERT)

- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)

- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
  
- [BigVSAN](https://github.com/sony/bigvsan)
