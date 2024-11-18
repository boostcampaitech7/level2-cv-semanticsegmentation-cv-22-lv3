import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

def loss_func_loader(config):
    loss_func_name = config.loss_func.name
    if loss_func_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_func_name == "DiceLoss":
        return smp.losses.DiceLoss(mode=config.loss_func.get("mode", "multilabel"))  # "binary", "multiclass", "multilabel" 모드 설정 가능
    elif loss_func_name == "JaccardLoss":
        return smp.losses.JaccardLoss(mode=config.loss_func.get("mode", "multilabel"))  # "binary", "multiclass", "multilabel" 모드 설정 가능
    elif loss_func_name == "FocalLoss":
        return smp.losses.FocalLoss(mode=config.loss_func.get("mode", "multilabel"), 
                                    alpha=config.loss_func.get("alpha", 0.25),
                                    gamma=config.loss_func.get("gamma", 0.2))
    else:
        raise ValueError(f"Unsupported loss_func: {loss_func_name}")

def optimizer_loader(config, model_parameters):
    optimizer_name = config.optimizer.name
    if optimizer_name == "Adam":
        return optim.Adam(params=model_parameters, lr=config.optimizer.lr, 
                          weight_decay=config.optimizer.weight_decay)
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=config.optimizer.lr, 
                         momentum=config.optimizer.get("momentum", 0.9), weight_decay=config.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

def lr_scheduler_loader(config, optimizer):
    scheduler_name = config.scheduler.name
    if scheduler_name == "CosineAnnealingLR":
         # 인자: optimizer, T_max(필수, 주기), eta_min(기본값: 0), last_epoch(기본값: -1)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.get("eta_min", 0),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )
    
    elif scheduler_name == "StepLR":
        # 인자: optimizer, step_size(필수, 학습률 감소 간격), gamma(기본값: 0.1), last_epoch(기본값: -1)
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.get("gamma", 0.1),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )

    elif scheduler_name == "ReduceLROnPlateau":
        # 인자: optimizer, mode(기본값: "min"), factor(기본값: 0.1), patience(기본값: 10), threshold, cooldown, etc.
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.get("mode", "min"),
            factor=config.scheduler.get("factor", 0.1),
            patience=config.scheduler.get("patience", 10),
            threshold=config.scheduler.get("threshold", 0.0001),
            cooldown=config.scheduler.get("cooldown", 0),
            min_lr=config.scheduler.get("min_lr", 0),
            eps=config.scheduler.get("eps", 1e-8)
        )

    elif scheduler_name == "ExponentialLR":
        # 인자: optimizer, gamma(필수), last_epoch(기본값: -1)
        return optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=config.scheduler.gamma,
            last_epoch=config.scheduler.get("last_epoch", -1)
        )

    elif scheduler_name == "MultiStepLR":
        # 인자: optimizer, milestones(필수, 학습률 감소 시점 리스트), gamma(기본값: 0.1), last_epoch(기본값: -1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.get("gamma", 0.1),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )

    elif scheduler_name == "CosineAnnealingWarmRestarts":
        # 인자: optimizer, T_0(필수, 처음 재시작 주기), T_mult(기본값: 1), eta_min(기본값: 0), last_epoch(기본값: -1)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.T_0,
            T_mult=config.scheduler.get("T_mult", 1),
            eta_min=config.scheduler.get("eta_min", 0),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )
    
    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")