import os
import pytz
import torch
import wandb
import datetime
from omegaconf import DictConfig
from data.dataloader import set_seed
from .validation import validation
from models.utils.models_utils import get_model_output, save_model


def train(model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          criterion: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler._LRScheduler, 
          config: DictConfig) -> None:
    '''
    summary:
        두 단계 학습 방식으로 모델 학습을 수행하는 함수입니다. 
        첫 번째 단계는 train 데이터셋으로 학습하고 val 데이터셋으로 검증합니다.
        두 번째 단계는 val 데이터셋으로 학습하고 train 데이터셋으로 검증합니다.

    args:
        model (torch.nn.Module): 학습할 모델 객체.
        train_loader (torch.utils.data.DataLoader): 학습 데이터 로더.
        val_loader (torch.utils.data.DataLoader): 검증 데이터 로더.
        criterion (torch.nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        scheduler (torch.optim.lr_scheduler._LRScheduler): 학습률 스케줄러.
        config (DictConfig): 학습 및 모델 설정 객체.

    return:
        None: 이 함수는 값을 반환하지 않습니다.
    '''

    print(f'max_epoch: {config.data.train.max_epoch}, valid & save_interval: {config.data.valid.interval}')
    print(f'Start training..')

    set_seed(config.seed) 
    kst = pytz.timezone('Asia/Seoul')

    best_dice = 0.0

    epochs_no_improve = 0
    patience = config.data.train.early_stopping_patience 
    delta = config.data.train.early_stopping_delta  

    os.makedirs(config.save.save_ckpt, exist_ok=True)

    model = model.cuda()

    for stage in range(1, 3):
        if stage == 1:
            stage_epoch = int(config.data.train.max_epoch)
            stage_trainloader = train_loader
            stage_valloader = val_loader
        else:
            stage_epoch = config.data.train.max_epoch - int(config.data.train.max_epoch * config.data.train.ratio) 
            stage_trainloader = val_loader
            stage_valloader = train_loader

        print(f'Stage_{stage}...')
        print(f'The number of train dataset : {len(stage_trainloader.dataset)}')
        print(f'The number of val dataset : {len(stage_valloader.dataset)}')

        for epoch in range(stage_epoch):
            model.train()
            epoch_loss = 0.0

            for step, loadered_data in enumerate(stage_trainloader):            
            
                if len(loadered_data) == 3 :
                    images = loadered_data[0]
                    masks = loadered_data[1]
                    weight_maps = loadered_data[2]
                
                else :
                    images = loadered_data[0]
                    masks = loadered_data[1]

                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)

                if config.loss_func.weight_map == True :
                    weight_maps = weight_maps.cuda(non_blocking=True)

                optimizer.zero_grad()
                outputs = get_model_output(model, images)

                if config.loss_func.weight_map == True :
                    loss = criterion(outputs, masks, weight_maps)
                else :
                    loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if (step + 1) % config.data.train.print_step == 0:
                    current_time = datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')
                    print(
                        f'{current_time} || '
                        f'Stage{stage} || Epoch [{epoch+1}/{stage_epoch}] | '
                        f'Step [{step+1}/{len(stage_trainloader)}] | '
                        f'Loss: {round(loss.item(),4)}'
                    )
                    wandb.log({f'Stage{stage} : train_loss' : loss.item(), 'Epoch' : epoch + 1 })

            scheduler.step()   

            if (epoch + 1) % config.data.valid.interval == 0:
                dice = validation(model, stage_valloader, config=config)  

                save_model(model, file_name=f'Stage{stage}_epoch_{epoch+1}_model', config=config)
                print(f'Save epoch {epoch+1} model in {config.save.save_ckpt}')

                if best_dice + delta <= dice:
                    print(f'Stage{stage} Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}')
                    print(f'Stage{stage} Save best model in {config.save.save_ckpt}')
                    best_dice = dice
                    save_model(model, file_name=f'{config.model.base_model}_best_model', config=config)

                else:
                    epochs_no_improve += 1
                    print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')
                    print(f'Stage{stage} performance at epoch: {epoch + 1}, {dice:.4f}, dice score decline : {(best_dice - dice):.4f}')

                    if epochs_no_improve >= patience:
                        print(f'Stage{stage} Early stopping triggered after {patience} epochs with no improvement.')
                        print(f'Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}')
                        break