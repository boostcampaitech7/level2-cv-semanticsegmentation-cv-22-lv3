import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from .custom_loss import CombinedWeightedLoss, TwoWayLoss, BCEDiceLoss


def loss_func_loader(config):
    loss_func_name = config.loss_func.name
    defaults_loss = config.loss_func_defaults

    if loss_func_name not in defaults_loss:
        raise ValueError(f"Unsupported loss_func: {loss_func_name}")
    
    # 기본 파라미터 불러오기
    loss_func_params = defaults_loss[loss_func_name]
    # 사용자 지정 파라미터로 재설정
    user_params = config.loss_func.get('params', {})
    loss_func_params.update(user_params)

    if loss_func_name == "BCEWithLogitsLoss":
        config.loss_func.weight_map = False
        return nn.BCEWithLogitsLoss()
    elif loss_func_name == "DiceLoss":
        config.loss_func.weight_map = False
        return smp.losses.DiceLoss(**loss_func_params)
    elif loss_func_name == "JaccardLoss":
        config.loss_func.weight_map = False
        return smp.losses.JaccardLoss(**loss_func_params)
    elif loss_func_name == "FocalLoss":
        config.loss_func.weight_map = False
        return smp.losses.FocalLoss(**loss_func_params)
    elif loss_func_name == "CombinedWeightedLoss":
        config.loss_func.weight_map = True
        return CombinedWeightedLoss(**loss_func_params)
    elif loss_func_name == "TwoWayLoss":
        config.loss_func.weight_map = True
        return TwoWayLoss(**loss_func_params)
    elif loss_func_name == "BCEDiceLoss":
        return BCEDiceLoss(**loss_func_params)
    else:
        raise ValueError(f"Unsupported loss_func: {loss_func_name}")


def optimizer_loader(config, model_parameters):
    optimizer_name = config.optimizer.name
    defaults_optimizer = config.optimizer_defaults

    if optimizer_name not in defaults_optimizer:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # 기본 파라미터 불러오기
    optimizer_params = defaults_optimizer[optimizer_name]
    # 사용자 지정 파라미터로 재설정
    user_params = config.optimizer.get('params', {})
    optimizer_params.update(user_params)
    
    if optimizer_name == "Adam":
        return optim.Adam(params=model_parameters, **optimizer_params)
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, **optimizer_params)
    elif optimizer_name == "AdamW":
        return optim.AdamW(params=model_parameters, **optimizer_params)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(params=model_parameters, **optimizer_params)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(params=model_parameters, **optimizer_params)
    elif optimizer_name == "Adadelta":
        return optim.Adadelta(params=model_parameters, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

def lr_scheduler_loader(config, optimizer):
    scheduler_name = config.scheduler.name
    defaults_scheduler = config.scheduler_defaults

    if scheduler_name not in defaults_scheduler:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    # 기본 파라미터 불러오기
    scheduler_params = defaults_scheduler[scheduler_name]
    # 사용자 지정 파라미터로 재설정
    user_params = config.scheduler.get('params', {})
    scheduler_params.update(user_params)

    if scheduler_name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    
    elif scheduler_name == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    elif scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,**scheduler_params)

    elif scheduler_name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)

    elif scheduler_name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)

    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    
    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")
    