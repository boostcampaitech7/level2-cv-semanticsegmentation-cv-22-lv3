import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp


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
        return nn.BCEWithLogitsLoss()
    elif loss_func_name == "DiceLoss":
        return smp.losses.DiceLoss(**loss_func_params)
    elif loss_func_name == "JaccardLoss":
        return smp.losses.JaccardLoss(**loss_func_params)
    elif loss_func_name == "FocalLoss":
        return smp.losses.FocalLoss(**loss_func_params)
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
    






import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt


def create_weight_map(mask, boundary_width=5, weight_inside=1.0, weight_boundary=2.0):
    '''
        Args :
            mask : 이진 마스크 (H, W)
            boundary_width : 강조할 경계의 너비
            weight_inside : 내부 픽셀의 가중치
            weight_boundary : 경계 픽셀의 가중치
    '''
    mask = mask.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    
    # 여러 번 침식하여 경계 너비 확대
    boundary = mask.copy()
    for _ in range(boundary_width):
        eroded = cv2.erode(boundary, kernel, iterations=1)
        boundary = boundary - eroded
    
    # 가중치 맵 초기화
    weight_map = np.ones_like(mask, dtype=np.float32) * weight_inside
    
    # 경계 픽셀에 높은 가중치 할당
    weight_map[boundary == 1] = weight_boundary
    
    return weight_map


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_inside=1.0, weight_boundary=2.0, smooth=1e-6):
        '''
            Args :
                weight_inside : 내부 픽셀의 가중치
                weight_boundary : 경계 픽셀의 가중치
                smooth : 분모가 0이 되는 것을 방지하기 위한 평활화 계수
        '''
        super(WeightedDiceLoss, self).__init__()
        self.weight_inside = weight_inside
        self.weight_boundary = weight_boundary
        self.smooth = smooth

    def forward(self, logits, targets):
        '''
            Args :
                logits : 모델의 출력 로짓 (N, C, H, W)
                targets : 실제 마스크 (N, C, H, W)
        '''
        # 시그모이드 활성화 적용하여 확률로 변환
        probs = torch.sigmoid(logits)

        N, C, H, W = probs.shape
        loss = 0.0

        # 각 클래스별로 손실 계산
        for c in range(C):
            probs_c = probs[:, c, :, :].contiguous().view(N, -1)  # (N, H*W)
            targets_c = targets[:, c, :, :].contiguous().view(N, -1)  # (N, H*W)

            # NumPy로 변환하여 가중치 맵 생성
            probs_np = probs_c.detach().cpu().numpy()
            targets_np = targets_c.detach().cpu().numpy()

            weight_maps = []
            for target in targets_np:
                weight_map = create_weight_map(target)
                weight_maps.append(weight_map)
            weight_maps = np.stack(weight_maps)
            weight_maps = torch.tensor(weight_maps, dtype=torch.float32).to(logits.device)

            # 텐서 평탄화
            weight_maps_flat = weight_maps.view(N, -1)  # (N, H*W)

            # 가중치를 적용한 교집합과 합집합 계산
            intersection = (probs_c * targets_c * weight_maps_flat).sum(dim=1)  # (N,)
            union = (probs_c * weight_maps_flat).sum(dim=1) + (targets_c * weight_maps_flat).sum(dim=1)  # (N,)

            # Dice 계수 계산
            dice = (2. * intersection + self.smooth) / (union + self.smooth)  # (N,)

            # 손실 누적 (평균)
            loss += (1 - dice).mean()

        # 클래스별 평균 손실 반환
        return loss / C