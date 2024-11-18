import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp


def loss_func_loader(config):
    loss_func_name = config.loss_func.name
    if loss_func_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_func_name == "DiceLoss":
        return smp.losses.DiceLoss(mode=config.loss_func.get("mode", "multilabel"))
    elif loss_func_name == "JaccardLoss":
        return smp.losses.JaccardLoss(mode=config.loss_func.get("mode", "multilabel"))
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
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.get("eta_min", 0),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )
    

    elif scheduler_name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.get("gamma", 0.1),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )


    elif scheduler_name == "ReduceLROnPlateau":
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
        return optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=config.scheduler.gamma,
            last_epoch=config.scheduler.get("last_epoch", -1)
        )


    elif scheduler_name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.get("gamma", 0.1),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )


    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.T_0,
            T_mult=config.scheduler.get("T_mult", 1),
            eta_min=config.scheduler.get("eta_min", 0),
            last_epoch=config.scheduler.get("last_epoch", -1)
        )
    

    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")