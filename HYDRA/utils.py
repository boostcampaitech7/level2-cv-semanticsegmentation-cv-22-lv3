# utils.py
import random
import numpy as np
import torch
import os
import torch.optim as optim
import torch.nn.functional as F

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None
    print("Lion optimizer not installed.")

def set_seed(seed):
    """
    재현성을 위해 난수 시드 설정.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice 계수를 계산.
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2. * intersection + smooth) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + smooth)

def save_model(model, saved_dir, file_name):
    """
    모델 가중치 저장.
    """
    try:
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)
        print(f"Model saved at {output_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        raise

def encode_mask_to_rle_gpu(mask):
    """
    GPU 상에서 마스크 RLE 인코딩.
    """
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
    """
    CPU 상에서 마스크 RLE 인코딩.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_optimizer(name, config, parameters):
    """
    config에 따른 옵티마이저 생성.
    """
    optimizer_name = name.lower()
    defaults = config.optimizer_defaults

    if optimizer_name not in defaults:
        raise ValueError(f"Unsupported optimizer: {name}")

    optimizer_params = defaults[optimizer_name].copy()
    user_params = config.optimizer.get('params', {})
    optimizer_params.update(user_params)

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
            raise ImportError("Lion optimizer not installed.")
        return Lion(parameters, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_scheduler(name, config, optimizer):
    """
    config에 따른 스케줄러 생성.
    """
    scheduler_name = name.lower()
    defaults = config.scheduler_defaults

    if scheduler_name not in defaults:
        raise ValueError(f"Unsupported scheduler: {name}")

    scheduler_params = defaults[scheduler_name].copy()
    user_params = config.scheduler.get('params', {})
    scheduler_params.update(user_params)

    if scheduler_name == 'step_lr':
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
