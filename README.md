# 🚀 프로젝트 소개
> Baone Segmentation을 통해 정확하게 뼈를 인식하여 의료 진단 및 치료 계획을 개발하는데 목적을 두고 있습니다.

<br>

# 💁🏼‍♂️💁‍♀️ Member 소개
| 이름       | 기여 내용 |
|------------|-----------|
| **김한별** | - 베이스라인 구축 및 증강 실험  <br>  |
| **손지형** | - 베이스라인 구축 및 해상도 실험 <br>|
| **유지환** | - 베이스라인 구축 및 모델 실험 <br>  |
| **정승민** | - 베이스라인 구축 및 가설실험 <br>  |
| **조현준** | - 베이스라인 구축 및 앙상블 실험 <br>  |
---

<br>

# 🤜 팀의 목표
- Git issue를 통해 일정 관리를 진행하자!
- 우리만의 Segmentation 베이스 라인을 구축하자!
- 결과를 시각화 하여 가설을 세우고 이를 검증하자! 
- 가설 검증시 회의를 통해서 의견을 적극 제시하자!

<br>

# 🖥️ 프로젝트 진행 환경

### 하드웨어
- **GPU**: NVIDIA Tesla V100-SXM2 32GB
  - **메모리**: 32GB


### 소프트웨어
- **Driver Version**: 535.161.08
- **CUDA Version**: 12.2
- **Python Version**: 3.10.13
- **Deep Learning Framework**: PyTorch, CUDA 지원 활성화

<br>


# 🗂️ 프로젝트 파일 구조 <br>
```bash
.
├── data
└── git_repository
    ├── configs
    │   └── base_config.yaml
    ├── requirements.txt
    └── src
        ├── Dataset
        │   ├── dataloader.py
        │   ├── dataset.py
        │   └── utils
        │       ├── splitdata.py
        │       └── transform.py
        ├── Model
        │   ├── model_loader.py
        │   ├── smp
        │   ├── torchvision
        │   └── utils
        │       ├── load_model.py
        │       ├── model_output.py
        │       └── modify_model.py
        ├── Train
        │   ├── loss
        │   │   ├── custom_loss.py
        │   │   └── loss_opt_sche.py
        │   ├── metrics
        │   │   └── metrics.py
        │   ├── trainer.py
        │   └── validation.py
        ├── Utils
        │   ├── combine_csv_predictions.py
        │   ├── config_utils.py
        │   ├── inference_utils.py
        │   ├── post_processing.py
        │   └── set_seed.py
        ├── Visualization
        │   ├── inference_visualization.py
        │   └── train_vis.py
        ├── __init__.py
        ├── ensemble.py
        ├── inference.py
        └── train.py
```


<br>

# 🧰 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

<br>

# 🦅 모델 학습 방법
*대회 규정상 baseline 코드는 .gitignore에 포함되어 현재 코드에 포함되어있지 않습니다*


### 학습 관련 인자
| 인자명                         | 타입      | 기본값                  | 설명 |
|-----------------------|-----------|-------------------------|------|
| `--config`            | `str`     | `'base_config.yaml'`               | 학습에 사용될 config 파일을 필요로 합니다. |
| `--model`            | `str`     | `'fcn_resnet50.yaml'`               | 학습에 사용될 모델 아키텍처의 config 파일을 필요로 합니다. |
| `--encoder`            | `str`     | `'None'`               | 학습에 사용될 모델의 Encoder를 설정할수 있습니다. 사용하지 않을 경우 기본 모델의 Encoder가 사용됩니다. |
| `--ckpt_path`           | `str`     | `None`                 | 사전 학습된 가중치의 경로를 지정하여 모델에 로드, 미입력시 ImageNet 사전 가중치가 적용됩니다. |



### Wandb 관련 인자
| 인자명                           | 타입      | 기본값                  | 설명 |
|-----------------------|-----------|-------------------------|------|
| `--project_name`        | `str`     | `'이름 미지정 프로젝트'` | Wandb에 사용할 프로젝트 이름 |
| `--run_name`            | `str`     | `None`                 | Wandb 실행(run) 이름. 설정하지 않으면 현재 시간을 기준으로 자동 생성됩니다. |




### 사용 예시
로컬 환경에서 돌리는 경우
```
python trian.py --config 'path/to/config' --model 'path/to/model' --encoder 'path/to/encdoer/' --project_name "Train Example Project" --run_name 'proejct test run'
```
nohup을 통해 GPU 서버 background에서 돌리는 경우 
```
nohup python trian.py --config 'path/to/config' --model 'path/to/model' --encoder 'path/to/encdoer/' --project_name "Train Example Project" --run_name 'proejct test run' > train_output.log 2>&1 &
```

<br>


# 🦇wandb

<div align="center">
  이미지 추후 추가 예정
</div>


---
wandb에서는 각 클래스별 Dice score, Train loss, Train Dice Score를 확인 할수 있으며, 학습시 첫 배치 랜덤 5개의 이미지를 Pred, GT, FP, FN, Overlay 이미지를 확인할 수 있습니다.
