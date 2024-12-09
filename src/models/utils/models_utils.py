import os.path as osp
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
from src.utils.config_utils import ConfigManager


def get_model_output(model: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
    '''
    summary:
        주어진 모델에 입력 데이터를 전달하고, 모델의 출력을 반환합니다. 
        출력 형식이 딕셔너리인 경우 'out' 키의 값을 반환하며, 그렇지 않으면 모델의 전체 출력을 반환합니다.

    args:
        model (torch.nn.Module): 입력 데이터를 처리할 PyTorch 모델 객체.
        input (torch.Tensor): 모델에 전달할 입력 데이터.

    return:
        torch.Tensor: 모델의 출력 데이터. 출력 형식이 딕셔너리라면 'out' 키의 값을 반환.
    '''
    outputs = model(input)['out'] if isinstance(model(input), dict) else model(input)
    return outputs


def save_model(model: torch.nn.Module, 
               file_name: str='fcn_resnet50_best_model', 
               config: OmegaConf=None) -> None:
    '''
    summary:
        주어진 모델을 지정된 파일 이름과 경로에 저장합니다. 
        모델의 라이브러리에 따라 저장 방식(torchvision 또는 smp)이 결정됩니다.

    args:
        model (torch.nn.Module): 저장할 모델 객체.
        file_name (str): 저장될 파일 이름. 기본값은 'fcn_resnet50_best_model'.
        config (OmegaConf): 모델 저장 설정을 포함하는 구성 객체.
    
    return:
        반환값이 없습니다.
    '''
    is_best = 'best' in file_name.lower()
    library = config.model.library

    file_name = f'{file_name}{".pt" if library == "torchvision" else ""}'
    save_ckpt = config.save.save_ckpt
    save_path = osp.join(save_ckpt, file_name)
    
    if is_best:
        ConfigManager.update_config(config=config, 
                                    key='inference.checkpoint_path', 
                                    value=save_path)
    
    if library == 'torchvision':
        torch.save(model.state_dict(), save_path)
    else:
        model.save_pretrained(save_path)
    
    return None


def load_model(model : torch.nn.Module, 
               checkpoint_path : str, 
               device : torch.device, 
               library : str) -> torch.nn.Module:
    '''
    summary :
        사전 훈련된 모델 체크포인트를 로드하고 지정된 디바이스에 배치합니다.

    args : 
        model : 초기화된 모델 객체
        checkpoint_path : 체크포인트 파일 경로
        device : 모델을 로드할 장치
        library : 사용하는 모델 라이브러리

    return :
        model : 로드된 모델
    '''
    if library == 'torchvision':
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        checkpoint = smp.from_pretrained(checkpoint_path)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    print(f'Checkpoint loaded successfully from: {checkpoint_path}')

    return model


def update_last_layer(model: torch.nn.Module, new_out_channels: int) -> torch.nn.Module:
    '''
    summary:
        모델의 분류기의 마지막 계층을 업데이트하여 새로운 출력 채널 수를 가지도록 수정합니다.
        이 함수는 기존 계층의 매개변수는 유지하면서, low_classifier, high_classifier 또는 Conv2D 계층을 업데이트합니다.

    args:
        model: 'classifier' 속성을 가지고 있으며, 경우에 따라 'aux_classifier'도 포함된 신경망 모델 객체.
        new_out_channels: 업데이트된 분류기 계층에서의 출력 채널 수.

    return:
        model : 업데이트된 분류기 계층을 포함한 수정된 모델.
    '''
    if hasattr(model, 'classifier'):
        classifier = model.classifier

        if hasattr(classifier, 'low_classifier') and hasattr(classifier, 'high_classifier'):
            old_low_classifier = classifier.low_classifier
            old_high_classifier = classifier.high_classifier

            new_low_classifier = nn.Conv2d(
                in_channels=old_low_classifier.in_channels,
                out_channels=new_out_channels,
                kernel_size=old_low_classifier.kernel_size,
                stride=old_low_classifier.stride,
                padding=old_low_classifier.padding,
                dilation=old_low_classifier.dilation,
                groups=old_low_classifier.groups,
                bias=(old_low_classifier.bias is not None),
                padding_mode=old_low_classifier.padding_mode
            )
            
            new_high_classifier = nn.Conv2d(
                in_channels=old_high_classifier.in_channels,
                out_channels=new_out_channels,
                kernel_size=old_high_classifier.kernel_size,
                stride=old_high_classifier.stride,
                padding=old_high_classifier.padding,
                dilation=old_high_classifier.dilation,
                groups=old_high_classifier.groups,
                bias=(old_high_classifier.bias is not None),
                padding_mode=old_high_classifier.padding_mode
            )
            
            classifier.low_classifier = new_low_classifier
            classifier.high_classifier = new_high_classifier

        elif hasattr(model, 'classifier') or hasattr(model, 'aux_classifier'):
            aux_classifier = model.aux_classifier

            for idx in reversed(range(len(classifier))):
                layer = classifier[idx]
                if isinstance(layer, nn.Conv2d):
                    old_layer = layer

                    if isinstance(old_layer, nn.Conv2d):
                        new_layer = nn.Conv2d(
                            in_channels=old_layer.in_channels,
                            out_channels=new_out_channels,
                            kernel_size=old_layer.kernel_size,
                            stride=old_layer.stride,
                            padding=old_layer.padding,
                            dilation=old_layer.dilation,
                            groups=old_layer.groups,
                            bias=(old_layer.bias is not None),
                            padding_mode=old_layer.padding_mode
                        )
                    
                    classifier[idx] = new_layer
                    break

            for idx in reversed(range(len(aux_classifier))):
                layer = aux_classifier[idx]
                if isinstance(layer, nn.Conv2d):
                    old_layer = layer

                    if isinstance(old_layer, nn.Conv2d):
                        new_layer = nn.Conv2d(
                            in_channels=old_layer.in_channels,
                            out_channels=new_out_channels,
                            kernel_size=old_layer.kernel_size,
                            stride=old_layer.stride,
                            padding=old_layer.padding,
                            dilation=old_layer.dilation,
                            groups=old_layer.groups,
                            bias=(old_layer.bias is not None),
                            padding_mode=old_layer.padding_mode
                        )
                    
                    aux_classifier[idx] = new_layer
                    break
    
    else:
        raise AttributeError('The model does not have a "classifier" attribute.')

    return model