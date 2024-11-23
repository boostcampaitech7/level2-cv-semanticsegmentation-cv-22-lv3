# import torch
# import torch.nn as nn
# from torchvision import models
# import segmentation_models_pytorch as smp
# from model.utils.modify_model import modify_lraspp_model, modify_model, modify_deeplabv3_model



# model_dict = {
#     # torchvision
#     'fcn_resnet50': models.segmentation.fcn_resnet50,
#     'fcn_resnet101': models.segmentation.fcn_resnet101,
#     'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
#     'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
#     'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large,
#     'deeplabv3_mobilenet_v3_large': models.segmentation.deeplabv3_mobilenet_v3_large,


#     # smp
#     'PAN': smp.PAN,
#     'FPN': smp.FPN,
#     'Unet': smp.Unet,
#     'MAnet': smp.MAnet,
#     'PSPNet': smp.PSPNet,
#     'Linknet': smp.Linknet,
#     'UPerNet': smp.UPerNet,
#     'DeepLabV3': smp.DeepLabV3,
#     'UnetPlusPlus': smp.UnetPlusPlus,
#     'DeepLabV3Plus': smp.DeepLabV3Plus,

#     }

# def model_loader(config):

#     model_config = config.model
#     library_name = model_config.library 
#     architecture_params = model_config.architecture
#     num_classes = len(config.data.classes)
    

#     base_model = architecture_params.base_model
#     if base_model not in model_dict:
#         raise ValueError(f"Unknown model name: {base_model}")


#     target_model = model_dict[base_model] 


#     if library_name == 'torchvision':
       
#         pretrained = architecture_params.get("pretrained", True)
#         weights = "DEFAULT" if pretrained else None
#         model = target_model(weights=weights)


#         if base_model == 'lraspp_mobilenet_v3_large':
#             model = modify_lraspp_model(model, num_classes)
#         elif 'deeplabv3' in base_model:
#             model = modify_deeplabv3_model(model, num_classes)
#         else:
#             model = modify_model(model, num_classes)

#         print(f"Loaded model: {base_model} with {num_classes} classes.")


#     elif library_name == 'smp':
#         model = target_model(**architecture_params)
    
    
#     else:
#         raise ValueError(f"Unknown library: {library_name}")
           

#     return model




import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from model.utils.modify_model import modify_lraspp_model, modify_model, modify_deeplabv3_model
import json
from safetensors.torch import load_file
import os

model_dict = {
    # torchvision
    'fcn_resnet50': models.segmentation.fcn_resnet50,
    'fcn_resnet101': models.segmentation.fcn_resnet101,
    'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
    'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
    'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large,
    'deeplabv3_mobilenet_v3_large': models.segmentation.deeplabv3_mobilenet_v3_large,

    # smp
    'PAN': smp.PAN,
    'FPN': smp.FPN,
    'Unet': smp.Unet,
    'MAnet': smp.MAnet,
    'PSPNet': smp.PSPNet,
    'Linknet': smp.Linknet,
    'UPerNet': smp.UPerNet,
    'DeepLabV3': smp.DeepLabV3,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
}

def model_loader(config):
    model_config = config.model
    library_name = model_config.library 
    architecture_params = model_config.architecture
    num_classes = len(config.data.classes)
    weights_dir = model_config.get('weights_dir', None)  # weights_dir가 존재하면 가져옴

    base_model = architecture_params.base_model
    if base_model not in model_dict:
        raise ValueError(f"Unknown model name: {base_model}")

    target_model = model_dict[base_model] 

    if library_name == 'torchvision':
        # torchvision 모델 처리
        pretrained = architecture_params.get("pretrained", True)
        weights = "DEFAULT" if pretrained else None
        model = target_model(weights=weights)

        if base_model == 'lraspp_mobilenet_v3_large':
            model = modify_lraspp_model(model, num_classes)
        elif 'deeplabv3' in base_model:
            model = modify_deeplabv3_model(model, num_classes)
        else:
            model = modify_model(model, num_classes)

        print(f"Loaded torchvision model: {base_model} with {num_classes} classes.")

    elif library_name == 'smp':
        # smp 모델 처리
        model = target_model(**architecture_params)
        print(f"Loaded SMP model: {base_model} with {num_classes} classes.")

    else:
        raise ValueError(f"Unknown library: {library_name}")

    # weights_dir가 제공되면 사용자 정의 가중치 로드
    if weights_dir:
        try:
            # config.json 경로 설정
            config_path = os.path.join(weights_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    weights_config = json.load(f)
                print(f"Loaded config from {config_path}.")
            else:
                print(f"No config.json found in {weights_dir}.")

            # model.safetensors 경로 설정
            weights_file = os.path.join(weights_dir, 'model.safetensors')
            if os.path.exists(weights_file):
                # safetensors 형식으로부터 state_dict 로드
                state_dict = load_file(weights_file)

                # 현재 모델의 state_dict 가져오기
                model_state_dict = model.state_dict()

                # 불일치하는 레이어 키 제거
                filtered_state_dict = {}
                for key in state_dict:
                    if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping loading parameter '{key}' due to size mismatch.")

                # 필터링된 state_dict 로드
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded custom weights from {weights_file}.")

                # 마지막 레이어(segmentation_head) 재초기화
                if library_name == 'smp' and hasattr(model, 'segmentation_head'):
                    # segmentation_head가 Sequential 또는 nn.ModuleList인 경우
                    if isinstance(model.segmentation_head, (nn.Sequential, nn.ModuleList)):
                        for layer in model.segmentation_head:
                            if isinstance(layer, nn.Conv2d):
                                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                                if layer.bias is not None:
                                    nn.init.constant_(layer.bias, 0)
                    elif isinstance(model.segmentation_head, nn.Conv2d):
                        nn.init.kaiming_normal_(model.segmentation_head.weight, mode='fan_out', nonlinearity='relu')
                        if model.segmentation_head.bias is not None:
                            nn.init.constant_(model.segmentation_head.bias, 0)
                    print("Reinitialized the final segmentation head layers.")
                else:
                    print("No segmentation_head found to reinitialize or library is not 'smp'.")
            else:
                raise FileNotFoundError(f"model.safetensors not found in {weights_dir}.")

        except Exception as e:
            raise ValueError(f"Error loading weights from {weights_dir}: {e}")

    return model