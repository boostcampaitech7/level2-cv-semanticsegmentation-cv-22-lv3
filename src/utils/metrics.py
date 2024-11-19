import torch
# import torch.nn.functional as F


# def dice_coef(preds, targets, num_classes, epsilon=1e-7):
#     preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()  
#     targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  

#     dims = (0, 2, 3)  
#     intersection = torch.sum(preds_one_hot * targets_one_hot, dims)
#     cardinality = torch.sum(preds_one_hot + targets_one_hot, dims)
#     dice = (2. * intersection + epsilon) / (cardinality + epsilon)

#     return dice  


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)