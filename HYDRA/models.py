# models.py
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig

def initialize_model(model_name, num_classes, pretrained=True, aux_loss=True,
                     encoder_name='nvidia/segformer-b4', encoder_weights=None, config=None):
    """
    주어진 model_name에 따라 모델을 초기화합니다.
    """
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

    all_models = {**torchvision_models, **smp_models, **transformer_models}

    if model_name not in all_models:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in smp_models:
        model_class = smp_models[model_name]
        model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=3,
            classes=num_classes
        )
    elif model_name in transformer_models:
        # Segformer
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
        # Torchvision models
        model = all_models[model_name](pretrained=pretrained, aux_loss=aux_loss)
        # Adjust classifier layers
        if model_name.startswith('fcn') or model_name.startswith('deeplabv3'):
            in_channels = model.classifier[4].in_channels
            model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        elif model_name == 'lraspp_mobilenet_v3_large':
            in_channels = model.classifier.low_classifier.in_channels
            model.classifier.low_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            in_channels = model.classifier.high_classifier.in_channels
            model.classifier.high_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        if aux_loss and hasattr(model, 'aux_classifier'):
            if model_name.startswith('fcn') or model_name.startswith('deeplabv3'):
                in_channels = model.aux_classifier[4].in_channels
                model.aux_classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            elif model_name == 'lraspp_mobilenet_v3_large':
                in_channels = model.aux_classifier.in_channels
                model.aux_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model
