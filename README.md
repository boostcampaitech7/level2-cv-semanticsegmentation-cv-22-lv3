# 🚀 프로젝트 소개
> 뼈 분할(Bone Segmentation)은 의료 진단 및 치료 계획 수립에서 중요한 역할을 하며, 딥러닝 기술을 활용해 정확한 분할을 목표로 합니다.
> 
**주요 응용 분야**
1. **질병 진단:** 골절, 변형 등 뼈 관련 문제를 정확히 파악해 적절한 치료 제공.
2. **수술 계획:** 뼈 구조 분석으로 필요한 수술 방법과 재료 선정.
3. **의료 장비 제작:** 인공 관절 및 임플란트 설계에 필요한 정보 제공.
4. **의료 교육:** 병태 및 부상에 대한 이해와 수술 기술 연습에 활용.
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
| **김한별** | - baseline 훈련 과정 구현, 모델 및 증강 실험  |
| **손지형** | - 베이스라인 구축 및 해상도, 데이터 증강 실험 |
| **유지환** | - Hydra baseline 구축, 모델/증강/스케줄러 실험 진행, 데이터 EDA 진행 |
| **정승민** | - 베이스라인 구축, 손등 관련 가설 실험 진행  |
| **조현준** | - 베이스라인 추론 과정 구현, 결과 시각화 및 앙상블  |

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
학습 결과는 'runs/날짜_모델명' 폴더에 체크포인트, 해당 config 파일이 자동 생성됩니다.
```bash
python train.py --config 'path/to/config' --model 'path/to/model' --encoder 'path/to/encoder' --project_name 'Name' --run_name 'Name'
```

추론시 mode, config 파일과 ckpt 파일의 경로를 필요로 합니다.
추론 결과는 results '폴더에 모델명_체크포인트명_날짜'로 자동 생성됩니다.
```bash
python inference.py --mode 'gpu' --config 'path/to/config'
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
김한별 : 2002bigstar@gmail.com  <br> 손지형 : sonji0988@gmail.com  <br> 유지환 : harwsare@yonsei.ac.kr  <br> 정승민 : taky0315@naver.com  <br> 조현준 : aaiss0927@gamil.com   <br>

