import torch
import wandb
import random
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils.metrics import dice_coef
from utils.set_seed import set_seed
from model.utils.model_output import get_model_output
from Dataset.visualization.train_vis import visualize_predictions, save_image_for_visualization


def validation(model, data_loader, config=None):
    epoch = config.data.train.max_epoch
    print(f'Start validation #{epoch:2d}')
    model.eval()
    dices = []

    preds_to_visualize = []
    masks_to_visualize = []

    with torch.no_grad():

        for _, loadered_data in tqdm(enumerate(data_loader), total=len(data_loader)):

            if len(loadered_data) == 3 :
                images, masks, weight_maps = loadered_data
                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                weight_maps = weight_maps.cuda(non_blocking=True)
            
            else :
                images, masks = loadered_data
                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)


            outputs = get_model_output(model, images)
            

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode=config.data.interpolate.bilinear)
            

            outputs = torch.sigmoid(outputs)
            thr = config.data.valid.threshold
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            preds_to_visualize, masks_to_visualize = save_image_for_visualization(
                config, masks, preds_to_visualize, outputs, masks_to_visualize
            )

            dice = dice_coef(outputs, masks)
            dices.append(dice)


    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12} : {d.item():.4f}"
        for c, d in zip(config.data.classes, dices_per_class)
    ]
    dice_str = '\n'.join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    wandb.log({
        "validation_avg_dice": avg_dice,
        "epoch": epoch
    })

    dice_dict = {f"validation_dice_{cls}": dices_per_class[idx].item() for idx, cls in enumerate(config.data.classes)}
    dice_dict['epoch'] = epoch
    wandb.log(dice_dict)

    if len(preds_to_visualize) > 0:
        figures = []
        for pred, mask in zip(preds_to_visualize, masks_to_visualize):
            fig = visualize_predictions(pred, mask)
            figures.append(wandb.Image(fig, caption=f"Epoch: {epoch}"))
        wandb.log({"validation_results": figures, "epoch": epoch})

    return avg_dice