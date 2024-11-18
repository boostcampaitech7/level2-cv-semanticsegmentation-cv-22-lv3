import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from model.utils.modify_model import modify_lraspp_model, modify_model, modify_deeplabv3_model


# 모델 딕셔너리
model_dict = {
    # Torchvision models
    'fcn_resnet50': models.segmentation.fcn_resnet50,
    'fcn_resnet101': models.segmentation.fcn_resnet101,
    'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
    'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
    'deeplabv3_mobilenet_v3_large': models.segmentation.deeplabv3_mobilenet_v3_large,
    'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large,

    # SMP models
    'Unet': smp.Unet,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
    'Linknet': smp.Linknet,
    'MAnet': smp.MAnet,
    'PAN': smp.PAN,
    'UPerNet': smp.UPerNet,
    }

def model_loader(config):
    """
    Config 파일을 읽어 모델을 정의합니다.
    """
    # 'model' 키 아래의 설정을 가져옵니다.
    model_config = config.model
    # 'library' 키의 값 (라이브러리 이름)
    library_name = model_config.library  # 'smp' 또는 'torchvision'
    # 'architecture' 키 아래의 모델 파라미터를 가져옵니다.
    architecture_params = model_config.architecture
    # data의 classes에서 num_classes를 가져옵니다.
    num_classes = len(config.data.classes)
    
    # 모델 가져오기
    base_model = architecture_params.base_model
    if base_model not in model_dict:
        raise ValueError(f"Unknown model name: {base_model}")

    # 모델 로드
    target_model = model_dict[base_model]  # 통일된 방식으로 모델 클래스 가져오기

    # torchvision 모델일 경우
    if library_name == 'torchvision':
        # Optional 값 처리
        pretrained = architecture_params.get("pretrained", True)
        weights = "DEFAULT" if pretrained else None
        # 모델 로드
        model = target_model(weights=weights)

        # 클래스 수 조정 (모델 수정)
        if base_model == 'lraspp_mobilenet_v3_large':
            model = modify_lraspp_model(model, num_classes)
        elif 'deeplabv3' in base_model:
            model = modify_deeplabv3_model(model, num_classes)
        else:
            model = modify_model(model, num_classes)

        print(f"Loaded model: {base_model} with {num_classes} classes.")

    # SMP 모델인 경우
    elif library_name == 'smp':
        # SMP 모델인 경우, 모델 파라미터를 그대로 전달하여 모델 초기화
        model = target_model(**architecture_params)
    
    # torchvision, SMP 모델이 아닌 경우 에러 발생
    else:
        raise ValueError(f"Unknown library: {library_name}")
           
    return model
