import torch


def dice_coef(y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    '''
    summary :
        Dice 계수를 계산합니다. 예측된 세그먼테이션 결과와 실제 세그먼테이션 마스크 사이의 유사성을 측정합니다.

    args : 
        y_true : 실제 세그먼테이션 마스크
        y_pred : 예측된 세그먼테이션 마스크

    return :
        Dice 계수 값
    '''
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)