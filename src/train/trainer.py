import os
import random
import numpy as np
from tqdm.auto import tqdm
import datetime
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple, Optional
import argparse
import torch
import torch.nn.functional as F
from utils.metrics import dice_coef
from utils.Dataset.visualization.train_vis import visualize_predictions
import wandb
import matplotlib.pyplot as plt
import io
import PIL.Image


def merge_config(base_config: str, model_config: str, encoder_config: Optional[str] = None, 
                save_config: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    
    base_config = OmegaConf.load(base_config)
    model_config = OmegaConf.load(model_config)

    # Conditionally merge encoder configuration
    if encoder_config is not None:
        encoder_config = OmegaConf.load(encoder_config)
        merged_config = OmegaConf.merge(base_config, model_config, encoder_config)
    else:
        merged_config = OmegaConf.merge(base_config, model_config)

    # Save configuration if save_config is provided
    if save_config:
        OmegaConf.save(merged_config, save_config)


    # Return merged configuration and save path
    return merged_config


def load_config(base_config: str, model_config: str, encoder_config: Optional[str] = None, 
                save_config: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    
    config = merge_config(base_config, model_config, encoder_config, save_config, save_dir)

    return config


def save_model(model, file_name='fcn_resnet50_best_model.pt', config=None):
    output_path = os.path.join(config.save_dir, file_name)
    # torch.save(model, output_path)
    # 모델내에 pickle이 lambda 함수가 있어 직렬화 하지 못해 에러가 나는 것을 발견하여 이를 수정.
    torch.save(model.state_dict(), output_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_model_output(model, library, input):
    if library == 'smp':
        return model(input)
    else:
        return model(input)['out']


def validation(epoch, model, data_loader, criterion, config=None):
    print(f'Start validation #{epoch:2d}')
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
        
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
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


def train(model, train_loader, val_loader, criterion, optimizer, config):
    print(f'max_epoch: {config.train.max_epoch},valid & save_interval: {config.val.interval}')
    print(f'Start training..')

    best_dice = 0.0
    
    epochs_no_improve = 0
    patience = config.train.early_stopping_patience 
    delta = config.train.early_stopping_delta  

    # 체크포인트 저장할 폴더 없을 경우 생성
    os.makedirs(config.save_dir, exist_ok=True)

    # set 2 stage train
    for stage in range(1, 3):
        if stage == 1 :
            stage_epoch = int(config.train.max_epoch * config.train.ratio)
            stage_trainloader = train_loader
            stage_valloader = val_loader
        else:
            stage_epoch = int(config.train.max_epoch * float(1 - config.train.ratio ))
            stage_trainloader = val_loader
            stage_valloader = train_loader


        print(f'Stage_{stage}...')
        print(f'The number of train dataset : {len(train_loader)}')
        print(f'The number of train dataset : {len(val_loader)}')
        for epoch in range(stage_epoch + 1):
            model.train()

            for step, (images, masks) in enumerate(stage_trainloader):            
                # gpu 연산을 위해 device 할당합니다.
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()
                
                # library 인자 수정하기
                outputs = get_model_output(model, config.model.library, images)

                # loss를 계산합니다.
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # step 주기에 따라 loss를 출력합니다.
                if (step + 1) % config.train.print_step == 0:
                    print(
                        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                        f'Stage{stage} : Epoch [{epoch+1}/{int(config.train.max_epoch * config.train.ratio)}], '
                        f'Step [{step+1}/{len(stage_trainloader)}], '
                        f'Loss: {round(loss.item(),4)}'
                    )

                    wandb.log({'Stage{stage} : train_loss' : loss.item(), 'Epoch' : epoch + 1 })
                
            # validation 주기에 따라 loss를 출력 및 checkpoint 저장하고 best model을 저장합니다.
            if (epoch + 1) % config.val.interval == 0:
                dice = validation(epoch + 1, model, stage_valloader, criterion, config=config)

                save_model(model, file_name=f'epoch_{epoch+1}_model.pt', config=config)
                print(f"Save epoch {epoch+1}model in {config.save_dir}")

                if best_dice < dice + delta:
                    print(f"Stage{stage} Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    print(f"Stage{stage} Save best model in {config.save_dir}")
                    best_dice = dice
                    epochs_no_improve = 0
                    
                    if config.model.library == 'torchvision':
                        save_model(model, file_name=f'{config.model.architecture.model_name}_best_model.pt', config=config)
                        best_model_filename = f'{config.model.architecture.model_name}_best_model.pt'

                    else:
                        save_model(model, file_name=f'{config.model.architecture.base_model}_best_model.pt', config=config)
                        best_model_filename = f'{config.model.architecture.base_model}_best_model.pt'

                    # save checkpoint in artifacts
                    best_model_path = os.path.join(config.save_dir, best_model_filename)
                    best_artifact = wandb.Artifact('best-model', type='model')
                    best_artifact.add_file(best_model_path)
                    wandb.log_artifact(best_artifact)


                else:
                    epochs_no_improve += 1
                    print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')

                    if epochs_no_improve >= patience :
                        print(f"Stage{stage} Early stopping triggered after {patience} epochs with no improvement.")
                        print(f"Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}")
                        continue
        
