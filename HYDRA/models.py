import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp  # SMP 임포트
from transformers import SegformerForSemanticSegmentation, SegformerConfig  # SegFormer 임포트

def initialize_model(model_name, num_classes, pretrained=True, aux_loss=True, encoder_name='nvidia/segformer-b4', encoder_weights=None, config=None):
    # 지원하는 모델 목록
    torchvision_models = {
        'fcn_resnet50': models.segmentation.fcn_resnet50,
        'fcn_resnet101': models.segmentation.fcn_resnet101,
        'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
        'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
        'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large
    }

    smp_models = {
        'unet': smp.Unet,
        'unet++': smp.UnetPlusPlus,
        'manet': smp.MAnet,
        'linknet': smp.Linknet,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3+': smp.DeepLabV3Plus,
        'pan': smp.PAN
    }

    transformer_models = {
        'segformer': SegformerForSemanticSegmentation
    }

    # 모든 모델을 합침
    all_models = {**torchvision_models, **smp_models, **transformer_models}

    if model_name not in all_models:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    if model_name in smp_models:
        # SMP 모델 초기화
        model_class = smp_models[model_name]
        model = model_class(
            encoder_name=encoder_name,             # 백본 모델
            encoder_weights=encoder_weights if pretrained else None,  # 사전 학습된 가중치
            in_channels=3,                         # 입력 채널 수
            classes=num_classes                    # 출력 클래스 수
        )
    elif model_name in transformer_models:
        # Transformer 기반 모델 초기화
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                id2label={str(i): label for i, label in enumerate(config.CLASSES)},
                label2id={label: str(i) for i, label in enumerate(config.CLASSES)},
            )
        else:
            config_model = SegformerConfig.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                id2label={str(i): label for i, label in enumerate(config.CLASSES)},
                label2id={label: str(i) for i, label in enumerate(config.CLASSES)},
            )
            model = SegformerForSemanticSegmentation(config_model)
    else:
        # torchvision 모델 초기화
        model = all_models[model_name](pretrained=pretrained, aux_loss=aux_loss)

        # 주 분류기(classifier) 조정
        if model_name.startswith('fcn') or model_name.startswith('deeplabv3'):
            in_channels = model.classifier[4].in_channels
            model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        elif model_name == 'lraspp_mobilenet_v3_large':
            in_channels = model.classifier.low_classifier.in_channels
            model.classifier.low_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            in_channels = model.classifier.high_classifier.in_channels
            model.classifier.high_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # 보조 분류기(auxiliary classifier) 조정
        if aux_loss and hasattr(model, 'aux_classifier'):
            if model_name.startswith('fcn') or model_name.startswith('deeplabv3'):
                in_channels = model.aux_classifier[4].in_channels
                model.aux_classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            elif model_name == 'lraspp_mobilenet_v3_large':
                in_channels = model.aux_classifier.in_channels
                model.aux_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model
