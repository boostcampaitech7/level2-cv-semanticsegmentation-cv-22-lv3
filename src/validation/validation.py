import torch
import random
import wandb
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils.metrics import dice_coef
from utils.set_seed import set_seed
from model.utils.model_output import get_model_output
from Dataset.visualization.train_vis import visualize_predictions


def validation(epoch, model, data_loader, criterion, config=None):
    print(f'Start validation #{epoch:2d}')
    set_seed(config.seed)
    model.eval()
    model = model.cuda()  
    dices = []
    total_loss = 0
    cnt = 0

    
    images_to_visualize = []
    preds_to_visualize = []
    masks_to_visualize = []

    with torch.no_grad():
        n_class = len(config.data.classes)
        total_loss = 0
        cnt = 0

        for _, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            
            
            outputs = get_model_output(model, config.model.library, images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
          
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
 
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).detach().cpu() 
            gt = masks.argmax(dim=1).detach().cpu()      
            
         
            dice = dice_coef(gt, preds, num_classes=n_class)  
            dices.append(dice)
                
   
            if len(images_to_visualize) < 5:
                num_needed = 5 - len(images_to_visualize)
                batch_size = images.size(0)
                num_to_take = min(num_needed, batch_size)
                images_to_visualize.extend(images[:num_to_take].cpu())
                preds_to_visualize.extend(preds[:num_to_take].cpu())
                masks_to_visualize.extend(gt[:num_to_take].cpu())


    avg_loss = total_loss / cnt
    dices = torch.stack(dices, dim=0)  
    dices_per_class = dices.mean(dim=0)  
    avg_dice = dices_per_class.mean().item()


    print(f'Epoch [{epoch}], Validation Loss: {avg_loss:.4f}, Average Dice Coefficient: {avg_dice:.4f}')
    for idx, cls in enumerate(config.data.classes):
        print(f"{cls:<12}: {dices_per_class[idx].item():.4f}")


    wandb.log({
        "validation_loss": avg_loss,
        "validation_avg_dice": avg_dice,
        "epoch": epoch
    })


    dice_dict = {f"validation_dice_{cls}": dices_per_class[idx].item() for idx, cls in enumerate(config.data.classes)}
    dice_dict['epoch'] = epoch
    wandb.log(dice_dict)

  
    if len(images_to_visualize) > 0:
        figures = visualize_predictions(images_to_visualize, preds_to_visualize, masks_to_visualize, max_visualize=5)


        if figures:
            wandb.log({"validation_results": figures, "epoch": epoch})

    return avg_dice