import torch
import torch.nn.functional as F


# def dice_coef(y_true, y_pred):
#     y_true_f = y_true.flatten(2)
#     y_pred_f = y_pred.flatten(2)
#     intersection = torch.sum(y_true_f * y_pred_f, -1)
    
#     eps = 0.0001
#     return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def dice_coef(preds, targets, num_classes, epsilon=1e-7):
    """
    Calculate Dice Coefficient for each class.

    Args:
        preds (torch.Tensor): Predicted masks of shape (batch_size, H, W).
        targets (torch.Tensor): Ground truth masks of shape (batch_size, H, W).
        num_classes (int): Number of classes.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Dice coefficients for each class. Shape: (num_classes,)
    """
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)

    dims = (0, 2, 3)  # Sum over batch, height, and width
    intersection = torch.sum(preds_one_hot * targets_one_hot, dims)
    cardinality = torch.sum(preds_one_hot + targets_one_hot, dims)
    dice = (2. * intersection + epsilon) / (cardinality + epsilon)

    return dice  # Shape: (num_classes,)