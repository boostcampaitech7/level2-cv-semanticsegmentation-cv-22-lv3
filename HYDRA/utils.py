# utils.py

import random
import numpy as np
import torch
import os
import torch.optim as optim
import torch.nn.functional as F

# Import Lion optimizer from the lion-pytorch package
try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None
    print("Lion optimizer is not installed. Install it via 'pip install lion-pytorch' to use it.")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2. * intersection + smooth) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + smooth)

def dice_coef_multiclass(preds, labels, num_classes, smooth=1e-7):
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(preds_one_hot * labels_one_hot, dim=(2, 3))
    preds_sum = torch.sum(preds_one_hot, dim=(2, 3))
    labels_sum = torch.sum(labels_one_hot, dim=(2, 3))
    dice = (2 * intersection + smooth) / (preds_sum + labels_sum + smooth)
    # 클래스별 Dice 계산 후 평균
    return dice.mean(dim=1)

def save_model(model, saved_dir, file_name):
    try:
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)
        print(f"Model saved at {output_path}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        raise


def encode_mask_to_rle_gpu(mask):
    mask = mask.to(torch.uint8)
    pixels = mask.flatten()
    zeros = torch.zeros(1, dtype=pixels.dtype, device=pixels.device)
    pixels = torch.cat([zeros, pixels, zeros])
    changes = (pixels[1:] != pixels[:-1]).nonzero().squeeze() + 1
    runs = changes.clone()
    runs[1::2] = runs[1::2] - runs[::2]
    runs = runs.cpu().numpy()
    return ' '.join(str(x) for x in runs)


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_optimizer(name, config, parameters):
    optimizer_name = name.lower()
    defaults = config.optimizer_defaults

    if optimizer_name not in defaults:
        raise ValueError(f"Unsupported optimizer: {name}")

    # Get default parameters
    optimizer_params = defaults[optimizer_name].copy()

    # Override with user-specified parameters
    user_params = config.optimizer.get('params', {})
    optimizer_params.update(user_params)

    # Initialize optimizer
    if optimizer_name == 'adam':
        return optim.Adam(parameters, **optimizer_params)
    elif optimizer_name == 'adamw':
        return optim.AdamW(parameters, **optimizer_params)
    elif optimizer_name == 'sgd':
        return optim.SGD(parameters, **optimizer_params)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, **optimizer_params)
    elif optimizer_name == 'lion':
        if Lion is None:
            raise ImportError("Lion optimizer is not installed. Install it via 'pip install lion-pytorch'.")
        return Lion(parameters, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_scheduler(name, config, optimizer):
    scheduler_name = name.lower()
    defaults = config.scheduler_defaults

    if scheduler_name not in defaults:
        raise ValueError(f"Unsupported scheduler: {name}")

    # Get default parameters
    scheduler_params = defaults[scheduler_name].copy()

    # Override with user-specified parameters
    user_params = config.scheduler.get('params', {})
    scheduler_params.update(user_params)

    # Initialize scheduler
    if scheduler_name == 'step_lr':
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
