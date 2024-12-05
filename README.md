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
- [📊 결과 및 시각화](#-결과-및-시각화)
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

# 📊 결과 및 시각화



# 📈 성능 평가



# 📜 라이선스

# 📞 문의
  •	이메일: taky0315@naver.com
	•	GitHub Issues: [링크](https://github.com/chungSungMin)
	•	팀 노션 페이지: Notion 링크

