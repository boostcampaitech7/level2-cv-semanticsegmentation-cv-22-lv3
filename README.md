# 🚀 프로젝트 소개
> **Bone Segmentation**을 통해 정확하게 뼈를 인식하여 의료 진단 및 치료 계획을 개발하는 데 목적을 두고 있습니다.

<br>

# 📋 목차
- [🚀 프로젝트 소개](#-프로젝트-소개)

- [💁🏼‍♂️💁‍♀️ 멤버 소개](#-멤버-소개)
- [🎯 팀의 목표](#-팀의-목표)
- [🖥️ 프로젝트 진행 환경](#️-프로젝트-진행-환경)
- [🗂️ 프로젝트 파일 구조](#️-프로젝트-파일-구조)
- [🧰 필요한 라이브러리 설치](#-필요한-라이브러리-설치)
- [🚀 모델 학습 방법](#-모델-학습-방법)
- [📈 성능 평가](#-성능-평가)
- [📜 라이선스](#-라이선스)
- [📞 문의](#-문의)

<br>

# 💁🏼‍♂️💁‍♀️ 멤버 소개
| 이름       | 기여 내용 |
|------------|-----------|
| **김한별** | - 베이스라인 구축 및 증강 실험  |
| **손지형** | - 베이스라인 구축 및 해상도 실험 |
| **유지환** | - 베이스라인 구축 및 모델 실험  |
| **정승민** | - 베이스라인 구축 및 가설 실험  |
| **조현준** | - 베이스라인 구축 및 앙상블 실험  |

---

<br>

# 🎯 팀의 목표
- **Git Issue**를 통해 일정 관리를 진행하자!
- 우리만의 **Segmentation 베이스라인**을 구축하자!
- 결과를 **시각화**하여 **가설을 세우고 검증**하자!
- **가설 검증 시 회의**를 통해 의견을 적극 제시하자!

<br>

# 🖥️ 프로젝트 진행 환경

### 하드웨어
- **GPU**: NVIDIA Tesla V100-SXM2 32GB
  - **메모리**: 32GB

### 소프트웨어
- **Driver Version**: 535.161.08
- **CUDA Version**: 12.2
- **Python Version**: 3.10.13
- **Deep Learning Framework**: PyTorch (CUDA 지원 활성화)

<br>

# 🗂️ 프로젝트 파일 구조

```bash
.
├── data
└── git_repository
    ├── configs
    │   └── base_config.yaml
    ├── requirements.txt
    └── src
        ├── Dataset
        │   ├── dataloader.py
        │   ├── dataset.py
        │   └── utils
        │       ├── splitdata.py
        │       └── transform.py
        ├── Model
        │   ├── model_loader.py
        │   ├── smp
        │   ├── torchvision
        │   └── utils
        │       ├── load_model.py
        │       ├── model_output.py
        │       └── modify_model.py
        ├── Train
        │   ├── loss
        │   │   ├── custom_loss.py
        │   │   └── loss_opt_sche.py
        │   ├── metrics
        │   │   └── metrics.py
        │   ├── trainer.py
        │   └── validation.py
        ├── Utils
        │   ├── combine_csv_predictions.py
        │   ├── config_utils.py
        │   ├── inference_utils.py
        │   ├── post_processing.py
        │   └── set_seed.py
        ├── Visualization
        │   ├── inference_visualization.py
        │   └── train_vis.py
        ├── __init__.py
        ├── ensemble.py
        ├── inference.py
        └── train.py
```


<br>

# 🧰 필요한 라이브러리 설치
```bash
pip install requirements.txt
```


<br>

# 🚀 모델 학습 방법

학습시 원하는 config 파일과 model, encoder 파일을 필요로 하며 추가로 Wandb 사용시 project_name 과 run_name을 필요로 합니다.
```bash
python train.py --config 'path/to/config' --model 'path/to/model' --encoder 'path/to/encoder' --project_name 'Name' --run_name 'Name'
```

추론시 mode, config 파일과 ckpt 파일의 경로를 필요로 합니다.
```bash
python inference.py --mode 'gpu' --config 'path/to/config' --checkpoint 'path/to/ckpt'
```

<br>

# 📈 성능 평가
모델 실험 | AdamW | CosineAnnealingLR
<br>
U-Net++
| Encoder         | Resolution | Epoch | Loss | Val Dice Score | LB Dice Score | pretrain |
| :-------------: | :--------: | :---: | :--: | :-------: | :------: | :------: |
| VGG19           | 512 x 512  |  100  | BCE  |  0.9610   |  0.9577  |  Imagenet  |
| EfficientNetB7  | 512 x 512  |  100  | BCE  |  0.9632   |  0.9563  |  Imagenet  |
| tu-HRNet        | 512 x 512  |  100  | Dice |  0.9629   |  0.9560  |  Imagenet  |

UperNet
| Encoder         | Resolution   | Epoch | Loss | Val Dice Score | LB Dice Score | pretrain |
| :-------------: | :----------: | :---: | :--: | :-------: | :------: | :------: |
| RegNet-120      | 1024 x 1024  |  100  | Dice |  0.9708   |  0.9682  |  Imagenet  |
| EfficientNetB7  | 1024 x 1024  |  100  | BCE  |  0.9684   |  0.9659  |  Imagenet  |
| EfficientNetB6  | 1024 x 1024  |  100  | BCE  |  0.9679   |  0.9650  |  Imagenet  |

UNet
| Encoder            | Resolution    | Epoch | Loss | Val Dice Score | LB Dice Score | pretrain |
| :----------------: | :-----------: | :---: | :--: | :-------: | :------: | :------: |
| ResNext101_32x8d   | 2048 x 2048   |  100  | Dice |  0.9734   |  0.9715  |  Imagenet  |
| EfficientNetB7     | 1024 x 1024   |  100  | Dice |  0.9713   |  0.9685  |  Imagenet  |
| EfficientNetB7     | 2048 x 2048   |  100  | Dice |  0.9698   |  0.9648  |  Imagenet  |


# 📞 문의
  •	이메일: taky0315@naver.com | sonji0988@gmail.com | 2002bigstar@gmail.com | aaiss0927@gamil.com | harwsare@yonsei.ac.kr <br>
	•	GitHub Issues: [링크](https://github.com/chungSungMin) <br>
	•	팀 노션 페이지: Notion 링크

