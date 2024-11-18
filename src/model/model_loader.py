import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from model.utils.modify_model import modify_lraspp_model, modify_model, modify_deeplabv3_model



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
    

    base_model = architecture_params.base_model
    if base_model not in model_dict:
        raise ValueError(f"Unknown model name: {base_model}")


    target_model = model_dict[base_model] 


    if library_name == 'torchvision':
       
        pretrained = architecture_params.get("pretrained", True)
        weights = "DEFAULT" if pretrained else None
        model = target_model(weights=weights)


        if base_model == 'lraspp_mobilenet_v3_large':
            model = modify_lraspp_model(model, num_classes)
        elif 'deeplabv3' in base_model:
            model = modify_deeplabv3_model(model, num_classes)
        else:
            model = modify_model(model, num_classes)

        print(f"Loaded model: {base_model} with {num_classes} classes.")


    elif library_name == 'smp':
        model = target_model(**architecture_params)
    
    
    else:
        raise ValueError(f"Unknown library: {library_name}")
           

    return model, base_model
