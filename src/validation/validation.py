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
    model = model.cuda()  # 모델을 루프 외부에서 한 번만 GPU로 이동

    dices = []
    total_loss = 0
    cnt = 0

    # 시각화를 위해 저장할 이미지, 예측 마스크, 실제 마스크 리스트
    images_to_visualize = []
    preds_to_visualize = []
    masks_to_visualize = []

    with torch.no_grad():
        n_class = len(config.data.classes)
        total_loss = 0
        cnt = 0

        for _, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            
            # 모델 출력 얻기
            outputs = get_model_output(model, config.model.library, images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # 예측 출력의 크기를 실제 마스크의 크기에 맞춥니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            # 손실 계산
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).detach().cpu()  # Shape: (batch_size, H, W)
            gt = masks.argmax(dim=1).detach().cpu()      # Shape: (batch_size, H, W)
            
            # Dice 계산
            dice = dice_coef(gt, preds, num_classes=n_class)  # Shape: (num_classes,)
            dices.append(dice)
                
            # 시각화를 위해 최대 5개의 이미지 저장
            if len(images_to_visualize) < 5:
                num_needed = 5 - len(images_to_visualize)
                batch_size = images.size(0)
                num_to_take = min(num_needed, batch_size)
                images_to_visualize.extend(images[:num_to_take].cpu())
                preds_to_visualize.extend(preds[:num_to_take].cpu())
                masks_to_visualize.extend(gt[:num_to_take].cpu())

    # 평균 손실 및 Dice 계수 계산
    avg_loss = total_loss / cnt
    dices = torch.stack(dices, dim=0)  # Shape: (num_batches, num_classes)
    dices_per_class = dices.mean(dim=0)  # Shape: (num_classes,)
    avg_dice = dices_per_class.mean().item()

    # 로그 출력
    print(f'Epoch [{epoch}], Validation Loss: {avg_loss:.4f}, Average Dice Coefficient: {avg_dice:.4f}')
    for idx, cls in enumerate(config.data.classes):
        print(f"{cls:<12}: {dices_per_class[idx].item():.4f}")

    # wandb 로깅
    wandb.log({
        "validation_loss": avg_loss,
        "validation_avg_dice": avg_dice,
        "epoch": epoch
    })

    # 클래스별 Dice 로깅
    dice_dict = {f"validation_dice_{cls}": dices_per_class[idx].item() for idx, cls in enumerate(config.data.classes)}
    dice_dict['epoch'] = epoch
    wandb.log(dice_dict)

    # 시각화 함수 호출
    if len(images_to_visualize) > 0:
        figures = visualize_predictions(images_to_visualize, preds_to_visualize, masks_to_visualize, max_visualize=5)
        # wandb에 시각화 결과 업로드
        if figures:
            wandb.log({"validation_results": figures, "epoch": epoch})

    return avg_dice