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

def load_config(base_config: str, model_config: str, encoder_config: Optional[str] = None, save_path: Optional[str] = None) -> str:
    # Load configurations
    base_config = OmegaConf.load(base_config)
    model_config = OmegaConf.load(model_config)

    # Conditionally merge encoder configuration
    if encoder_config is not None:
        encoder_config = OmegaConf.load(encoder_config)
        merged_config = OmegaConf.merge(base_config, model_config, encoder_config)
    else:
        merged_config = OmegaConf.merge(base_config, model_config)

    # Save configuration if save_path is provided
    if save_path:
        OmegaConf.save(merged_config, save_path)

    # Return merged configuration and save path
    return merged_config

def save_model(model, file_name='fcn_resnet50_best_model.pt', config=None):
    output_path = os.path.join(config.save_dir, file_name)
    torch.save(model, output_path)


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

# valid set test
def validation(epoch, model, data_loader, criterion, config=None):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(config.data.classes)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            # library 인자 수정하기
            outputs = get_model_output(model, config.model.library, images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > config.val.threshold).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(config.data.classes, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

# trainer
def train(model, train_loader, val_loader, criterion, optimizer, config):
    print(f'max_epoch: {config.train.max_epoch}, valid & save_interval: {config.val.interval}')
    print(f'Start training..')

    config = config

    
    n_class = len(config.data.classes)
    best_dice = 0.

    # 체크포인트 저장할 폴더 없을 경우 생성
    os.makedirs(config.save_dir, exist_ok=True)
    
    for epoch in range(config.train.max_epoch):
        model.train()

        for step, (images, masks) in enumerate(train_loader):            
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
                    f'Epoch [{epoch+1}/{config.train.max_epoch}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력 및 checkpoint 저장하고 best model을 저장합니다.
        if (epoch + 1) % config.val.interval == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, config=config)

            save_model(model, file_name=f'epoch_{epoch+1}_model.pt', config=config)
            print(f"Save epoch {epoch+1}model in {config.save_dir}")

            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save best model in {config.save_dir}")
                best_dice = dice
                # library에 따라 모델 이름 다른 key값 반영
                if config.model.architecture.model_name:
                    save_model(model, file_name=f'{config.model.architecture.model_name}_best_model.pt', config=config)
                else:
                    save_model(model, file_name=f'{config.model.architecture.base_model}_best_model.pt', config=config)




