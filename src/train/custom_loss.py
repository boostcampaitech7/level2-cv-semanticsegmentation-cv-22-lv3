import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class WeightedFocalLoss(nn.Module):
    '''
    summary: 
        가중치와 Focal Loss를 결합한 손실 함수 클래스. 클래스 불균형 문제에 적합하며, 
        입력된 알파(alpha) 및 감마(gamma) 값을 활용하여 Focal Loss를 계산합니다.

    args:
        alpha (float | list | tuple | None): 클래스별 가중치 값. 
                                             None이면 모든 클래스의 가중치를 1로 설정합니다.
        gamma (float): Focal Loss의 감마 값. 기본값은 2.0.
        smooth (float): 안정성을 위한 작은 값. 기본값은 1e-6.
        reduction (str): 결과 손실값의 축소 방법 ('mean', 'sum', 'none'). 기본값은 'mean'.
    '''

    def __init__(self, alpha=None, gamma=2.0, smooth=1e-6, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([alpha] * 29, dtype=torch.float32)  
        elif alpha is None:
            self.alpha = torch.ones(29, dtype=torch.float32)  
        else:
            raise TypeError('alpha must be float, list, tuple, or None')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weight_maps: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            모델의 출력(logits)과 실제 레이블(targets)을 사용해 Focal Loss를 계산합니다.
            선택적으로 가중치 맵(weight_maps)을 적용할 수 있습니다.

        args:
            logits (torch.Tensor): 모델의 출력 값. 
                                   크기는 (batch_size, num_classes, height, width).
            targets (torch.Tensor): 실제 레이블. 
                                    크기는 (batch_size, num_classes, height, width).
            weight_maps (torch.Tensor): 픽셀 가중치 맵. 
                                        크기는 (batch_size, num_classes, height, width) 또는 None.

        return:
            torch.Tensor: 계산된 Focal Loss 값. 축소 방법에 따라 단일 스칼라 값 또는 텐서를 반환합니다.
        '''
        device = logits.device
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(device).view(1, -1, 1, 1)

        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, 
                                                      reduction='none')
        probs = torch.sigmoid(logits)
        focal_factor = (1-probs) ** self.gamma 

        if self.alpha is not None:
            BCE_loss = self.alpha * BCE_loss 
        focal_loss = focal_factor * BCE_loss 

        if weight_maps is not None:
            focal_loss = focal_loss * weight_maps 

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  
        

class WeightedBCEWithLogitsLoss(nn.Module):
    '''
    summary: 
        가중치를 적용한 BCEWithLogits 손실 함수 클래스입니다. 
        픽셀 단위 가중치 맵(weight_maps)을 활용하여 손실 값을 조정합니다.

    args:
        smooth (float): 안정성을 위한 작은 값. 기본값은 1e-6.
    '''
    def __init__(self, smooth=1e-6):

        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 개별 손실 계산을 위해 reduction='none' 사용

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weight_maps: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            모델의 출력(logits)과 실제 레이블(targets)을 기반으로 BCEWithLogits 손실을 계산하고,
            가중치 맵(weight_maps)을 적용합니다.

        args:
            logits (torch.Tensor): 모델의 출력 값.
            targets (torch.Tensor): 실제 레이블.
            weight_maps (torch.Tensor): 픽셀 가중치 맵.

        return:
            torch.Tensor: 계산된 가중 BCEWithLogits 손실 값. 평균 손실 값을 반환합니다.
        '''
        loss = self.bce(logits, targets)        
        weighted_loss = loss * weight_maps        
        return weighted_loss.mean()


class WeightedDiceLoss(nn.Module):
    '''
    summary: 
        가중치를 적용한 Dice 손실 함수 클래스입니다. 
        픽셀 단위 가중치 맵(weight_maps)을 활용하여 Dice 손실을 계산합니다.

    args:
        smooth (float): 안정성을 위한 작은 값. 기본값은 1e-6.
    '''
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weight_maps: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            모델의 출력(logits)과 실제 레이블(targets)을 기반으로 Dice 손실을 계산하고,
            가중치 맵(weight_maps)을 적용합니다.

        args:
            logits (torch.Tensor): 모델의 출력 값.
            targets (torch.Tensor): 실제 레이블.
            weight_maps (torch.Tensor): 픽셀 가중치 맵.

        return:
            torch.Tensor: 계산된 가중 Dice 손실 값. 평균 손실 값을 반환합니다.
        '''
        probs = torch.sigmoid(logits)
        intersection = (probs*targets*weight_maps).sum(dim=(2,3))
        union = (probs*weight_maps).sum(dim=(2,3)) + (targets*weight_maps).sum(dim=(2,3))

        dice = (2.*intersection + self.smooth) / (union+self.smooth)
        loss = 1 - dice
        return loss.mean()


# 위에서 정의한 3가지 loss를 이용하여 hybrid loss 정의

class CombinedWeightedLoss(nn.Module):
    '''
    summary:
        Dice Loss, BCEWithLogits Loss, Focal Loss를 가중치 기반으로 결합한 손실 함수 클래스입니다. 
        내부 영역과 경계 영역에 각각 다른 가중치를 부여하여 손실을 계산합니다.

    args:
        dice_weight (float): Dice 손실에 부여할 가중치. 기본값은 1.0.
        bce_weight (float): BCEWithLogits 손실에 부여할 가중치. 기본값은 1.0.
        focal_weight (float): Focal 손실에 부여할 가중치. 기본값은 1.0.
        alpha (float | list | tuple | None): Focal Loss에서 클래스 가중치 값. 
                                             None이면 모든 클래스 가중치를 1로 설정.
        gamma (float): Focal Loss의 감마 값. 기본값은 2.0.
        weight_inside (float): 내부 영역에 부여할 가중치. 기본값은 1.0.
        weight_boundary (float): 경계 영역에 부여할 가중치. 기본값은 2.0.
        smooth (float): 안정성을 위한 작은 값. 기본값은 1e-6.
    '''
    def __init__(self, dice_weight=1.0, bce_weight=1.0, focal_weight=1.0, 
                 alpha=None, gamma=2.0, weight_inside=1.0, weight_boundary=2.0, smooth=1e-6):
        
        super(CombinedWeightedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.weight_inside = weight_inside
        self.weight_boundary = weight_boundary
        self.smooth = smooth

        self.dice = WeightedDiceLoss(smooth=smooth)
        self.bce = WeightedBCEWithLogitsLoss(smooth=smooth)
        self.focal = WeightedFocalLoss(alpha=alpha, gamma=gamma, smooth=smooth, reduction='mean')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weight_maps: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            모델의 출력(logits)과 실제 레이블(targets)을 기반으로 Dice, BCEWithLogits, Focal Loss를 계산하고, 
            각 손실에 부여된 가중치를 반영하여 총 손실 값을 반환합니다.

        args:
            logits (torch.Tensor): 모델의 출력 값.
            targets (torch.Tensor): 실제 레이블.
            weight_maps (torch.Tensor): 픽셀 가중치 맵.

        return:
            torch.Tensor: 계산된 총 손실 값. 스칼라 값을 반환합니다.
        '''
        dice_loss = self.dice(logits, targets, weight_maps)
        bce_loss = self.bce(logits, targets, weight_maps)
        focal_loss = self.focal(logits, targets, weight_maps)

        total_loss = (self.dice_weight*dice_loss 
                      + self.bce_weight*bce_loss 
                      + self.focal_weight*focal_loss)        
        return total_loss
    

nINF = -100

class TwoWayLoss(nn.Module):
    '''
    summary:
        TwoWayLoss는 양방향 손실을 계산하는 클래스입니다. 
        양성(positive)과 음성(negative) 샘플에 대해 
        서로 다른 온도 스케일(Tp, Tn)을 적용하여 손실을 계산합니다.

    args:
        Tp (float): 양성 샘플에 대한 온도 스케일. 기본값은 4.0.
        Tn (float): 음성 샘플에 대한 온도 스케일. 기본값은 1.0.
    '''
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            입력된 logits(x)과 레이블(y)을 기반으로 양성 및 음성 손실을 계산하고, 
            이를 결합하여 최종 손실을 반환합니다.

        args:
            x (torch.Tensor): 모델의 출력 값(logits). 크기는 (batch_size, num_classes).
            y (torch.Tensor): 실제 레이블. 크기는 (batch_size, num_classes).

        return:
            torch.Tensor: 계산된 손실 값. 스칼라 값을 반환합니다.
        '''
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return F.softplus(nlogit_class + plogit_class).mean() + \
                F.softplus(nlogit_sample + plogit_sample).mean()


class BCEDiceLoss(nn.Module):
    '''
    summary:
        BCEWithLogits 손실과 Dice 손실을 결합한 손실 함수 클래스입니다. 
        주로 세그멘테이션 작업에 사용되며 두 손실의 합을 최종 손실로 반환합니다.

    args:
        dice_mode (str): Dice Loss의 계산 모드. 기본값은 'multilabel'.
    '''
    def __init__(self, dice_mode='multilabel'):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  
        self.dice = smp.losses.DiceLoss(mode=dice_mode) 

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        summary:
            입력된 logits과 targets를 기반으로 BCEWithLogits 손실과 Dice 손실을 계산하고, 
            이를 결합하여 최종 손실 값을 반환합니다.

        args:
            logits (torch.Tensor): 모델의 출력 값.
            targets (torch.Tensor): 실제 레이블.

        return:
            torch.Tensor: 계산된 총 손실 값. 스칼라 값을 반환합니다.
        '''
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)        

        total_loss = bce_loss + dice_loss
        return total_loss    