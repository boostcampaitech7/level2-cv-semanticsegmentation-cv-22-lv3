import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


''' hybrid loss를 설정하기 위해서 총 3개의 loss를 직접 구현
    - WeightedFocalLoss
    - WeightedBCEWithLogitsLoss
    - WeightedDiceLoss

    위의 3가지 loss를 모두 고려해서 하나의 loss값을 전달한다
'''

class WeightedFocalLoss(nn.Module):
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
            raise TypeError("alpha must be float, list, tuple, or None")

    def forward(self, logits, targets, weight_maps):
        device = logits.device
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(device).view(1, -1, 1, 1)  


        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  


        probs = torch.sigmoid(logits) 


        focal_factor = (1 - probs) ** self.gamma 


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
    def __init__(self, smooth=1e-6):

        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 개별 손실 계산을 위해 reduction='none' 사용

    def forward(self, logits, targets, weight_maps):

        loss = self.bce(logits, targets)
        
        weighted_loss = loss * weight_maps
        
        return weighted_loss.mean()


class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):

        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weight_maps):
        probs = torch.sigmoid(logits)

        intersection = (probs * targets * weight_maps).sum(dim=(2,3))
        union = (probs * weight_maps).sum(dim=(2,3)) + (targets * weight_maps).sum(dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice
        return loss.mean()


''' 
위에서 정의한 3가지 loss를 이용하여 hybrid loss 정의

다른 loss들과 다르게 forward 과정에서 weight_maps를 필요로 하기에
trainer.py과 validation.py에서 loss를 계산할 때 따로 처리가 필요해보인다.

그 계산 과정은 아래의 코드에서 진행된다. 
loss = criterion(outputs, masks, weight_maps)
'''

class CombinedWeightedLoss(nn.Module):
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

    def forward(self, logits, targets, weight_maps):
        dice_loss = self.dice(logits, targets, weight_maps)
        bce_loss = self.bce(logits, targets, weight_maps)
        focal_loss = self.focal(logits, targets, weight_maps)

        total_loss = (self.dice_weight * dice_loss) + \
                     (self.bce_weight * bce_loss) + \
                     (self.focal_weight * focal_loss)
        return total_loss
    

nINF = -100

class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
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
    def __init__(self, dice_mode="multilabel"):

        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  
        self.dice = smp.losses.DiceLoss(mode=dice_mode) 

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)

        dice_loss = self.dice(logits, targets)
        

        total_loss = bce_loss + dice_loss
        return total_loss
    