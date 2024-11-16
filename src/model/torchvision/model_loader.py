import torch
import torch.nn as nn
from torchvision import models
from model.utils.modify_model import modify_lraspp_model, modify_model, modify_deeplabv3_model


# 모델 딕셔너리
model_dict = {
    'fcn_resnet50': models.segmentation.fcn_resnet50,
    'fcn_resnet101': models.segmentation.fcn_resnet101,
    'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
    'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
    'deeplabv3_mobilenet_v3_large': models.segmentation.deeplabv3_mobilenet_v3_large,
    'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large,
}

def model_loader(config):
    """
    Config 파일을 읽어 모델을 정의합니다.
    """
    
    # Config 값 가져오기
    library = config.library  # 'library' 키의 값
    model_name = config.model_name  # 'model_name' 키의 값
    num_classes = len(config.data.classes)  # 'data.classes' 키의 값들의 길이

    # Optional 값 처리
    pretrained = getattr(config, 'pretrained', True)  # default: True
    weights = "DEFAULT" if pretrained else None

    # torchvision 모델일 경우
    if library == 'torchvision':
        if model_name not in model_dict:
            raise ValueError(f"Unknown model name : {model_name}")
        
        # 모델 로드
        model = model_dict[model_name](weights=weights)

        # 모델 수정
        if model_name == 'lraspp_mobilenet_v3_large':
            model = modify_lraspp_model(model, num_classes)
        elif 'deeplabv3' in model_name:
            model = modify_deeplabv3_model(model, num_classes)
        else:
            model = modify_model(model, num_classes)

    
    else:
        raise ValueError(f"Unknown library: {library}")

    print(f"Loaded model: {model_name} with {num_classes} classes.")
    return model
