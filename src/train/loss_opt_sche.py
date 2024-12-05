from typing import Generator
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf
from .custom_loss import CombinedWeightedLoss, TwoWayLoss, BCEDiceLoss


def loss_func_loader(config: OmegaConf) -> nn.Module:
    '''
    summary:
        주어진 설정(config)을 기반으로 손실 함수(loss function)를 로드하여 반환합니다. 
        지원되지 않는 손실 함수 요청 시 예외를 발생시킵니다.

    args:
        config (OmegaConf): 손실 함수 이름과 매개변수를 포함하는 설정 객체.

    return:
        nn.Module: 구성된 손실 함수 모듈 객체.
    '''
    loss_func_name = config.loss_func.name
    defaults_loss = config.loss_func_defaults

    if loss_func_name not in defaults_loss:
        raise ValueError(f"Unsupported loss_func: {loss_func_name}")

    loss_func_params = defaults_loss[loss_func_name]

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


def optimizer_loader(config: OmegaConf, model_parameters: Generator) -> optim.Optimizer:
    '''
    summary:
        주어진 설정(config)과 모델의 파라미터를 기반으로 최적화 알고리즘(optimizer)을 로드하여 반환합니다.
        지원되지 않는 옵티마이저 요청 시 예외를 발생시킵니다.

    args:
        config (OmegaConf): 옵티마이저 이름과 매개변수를 포함하는 설정 객체.
        model_parameters (Generator): 모델의 학습 가능한 파라미터.

    return:
        optim.Optimizer: 구성된 옵티마이저 객체.
    '''
    optimizer_name = config.optimizer.name
    defaults_optimizer = config.optimizer_defaults

    if optimizer_name not in defaults_optimizer:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")    

    optimizer_params = defaults_optimizer[optimizer_name]
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
    

def lr_scheduler_loader(config: OmegaConf, 
                        optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    '''
    summary:
        주어진 설정(config)과 옵티마이저를 기반으로 
        학습률 스케줄러(learning rate scheduler)를 로드하여 반환합니다.
        지원되지 않는 스케줄러 요청 시 예외를 발생시킵니다.

    args:
        config (OmegaConf): 스케줄러 이름과 매개변수를 포함하는 설정 객체.
        optimizer (optim.Optimizer): 학습률 스케줄러가 적용될 옵티마이저.

    return:
        optim.lr_scheduler._LRScheduler: 구성된 학습률 스케줄러 객체.
    '''
    scheduler_name = config.scheduler.name
    defaults_scheduler = config.scheduler_defaults

    if scheduler_name not in defaults_scheduler:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")    

    scheduler_params = defaults_scheduler[scheduler_name]

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