import os
import datetime
import pytz
import pytz
import torch
import wandb
from utils.set_seed import set_seed
from validation.validation import validation
from model.utils.model_output import get_model_output


def save_model(model, file_name='fcn_resnet50_best_model', config=None):
    library = config.model.library
    file_name = f"{file_name}{'.pt' if library == 'torchvision' else ''}"
    save_ckpt = config.save.save_ckpt
    save_path = os.path.join(save_ckpt, file_name)

    if library == 'torchvision':
        torch.save(model.state_dict(), save_path)
    else:
        model.save_pretrained(save_path)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config) -> None:
    '''
        summary : We train the model using a two-stage approach. 
            No. 1: Learn with train and verify with val
            No. 2: Learn with val and verify with train

        args : model
            train_loader
            val_loader
            criterion
            optimizer
            scheduler
            config
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

            # for step, (images, masks, weight_maps) in enumerate(stage_trainloader):
            #     images = images.cuda(non_blocking=True)
            #     masks = masks.cuda(non_blocking=True)
            #     weight_maps = weight_maps.cuda(non_blocking=True)
            
            for step, (images, masks) in enumerate(stage_trainloader):
                images, masks = images.cuda(), masks.cuda()

                optimizer.zero_grad()
                outputs = get_model_output(model, images)

                # loss = criterion(outputs, masks, weight_maps)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if (step + 1) % config.data.train.print_step == 0:
                    current_time = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f'{current_time} || '
                        f'Stage{stage} || Epoch [{epoch+1}/{stage_epoch}] | '
                        f'Step [{step+1}/{len(stage_trainloader)}] | '
                        f'Loss: {round(loss.item(),4)}'
                    )
                    wandb.log({f'Stage{stage} : train_loss' : loss.item(), 'Epoch' : epoch + 1 })

            scheduler.step()   

            if (epoch + 1) % config.data.valid.interval == 0:
                dice = validation(epoch + 1, model, stage_valloader, criterion, config=config)  

                save_model(model, file_name=f'epoch_{epoch+1}_model', config=config)
                print(f"Save epoch {epoch+1} model in {config.save.save_ckpt}")

                if best_dice + delta <= dice:
                    print(f"Stage{stage} Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    print(f"Stage{stage} Save best model in {config.save.save_ckpt}")
                    best_dice = dice
                    save_model(model, file_name=f'{config.model.architecture.base_model}_best_model', config=config)

                else:
                    epochs_no_improve += 1
                    print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')
                    print(f"Stage{stage} performance at epoch: {epoch + 1}, {dice:.4f}, dice score decline : {(best_dice - dice):.4f}")

                    if epochs_no_improve >= patience:
                        print(f"Stage{stage} Early stopping triggered after {patience} epochs with no improvement.")
                        print(f"Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}")
                        break


            '''
                아래 주석을 모두 풀고 모델을 돌리는 경우에는 지정해둔 20 ~ 27 까지의 index 값을 갖는 가중치 맵이 생성됩니다.
            '''

            # if epoch == 0:
            #     sample_images, sample_masks, sample_weight_maps = next(iter(stage_trainloader))
            #     sample_images = sample_images.cuda(non_blocking=True)
            #     sample_masks = sample_masks.cuda(non_blocking=True)
            #     sample_weight_maps = sample_weight_maps.cuda(non_blocking=True)

            #     with torch.no_grad():
            #         sample_outputs = model(sample_images)
            #         sample_probs = torch.sigmoid(sample_outputs).detach().cpu().numpy()
            #         sample_masks_np = sample_masks.detach().cpu().numpy()
            #         sample_weight_maps_np = sample_weight_maps.detach().cpu().numpy()

            #     selected_classes = [20, 21, 22, 23, 24, 25, 26, 27] 

            #     for class_idx in selected_classes:
            #         mask = sample_masks_np[0, class_idx, :, :]  
            #         weight_map = sample_weight_maps_np[0, class_idx, :, :] 

            #         visualize_weight_map(mask, weight_map, class_idx, epoch + 1, stage)





# def visualize_weight_map(mask, weight_map, class_idx, epoch, stage):
#     save_dir = '/data/ephemeral/home/JSM_TEST_vis'
    
#     # 디렉토리가 존재하지 않으면 생성
#     os.makedirs(save_dir, exist_ok=True)
    
#     # mask와 weight_map이 numpy 배열이라고 가정
#     # mask를 PIL 이미지로 변환 (이진 마스크일 경우)
#     mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
    
#     # weight_map을 정규화하여 0-255 범위로 변환
#     weight_map_normalized = (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))
#     weight_map_uint8 = (weight_map_normalized * 255).astype(np.uint8)
#     weight_map_image = Image.fromarray(weight_map_uint8).convert('RGB')
    
#     # overlay 이미지 생성
#     overlay = Image.blend(mask_image.convert('RGB'), weight_map_image, alpha=0.5)
    
#     # 파일명 생성
#     filename_mask = f'stage_{stage}_epoch_{epoch}_class_{class_idx}_mask.png'
#     filename_weight_map = f'stage_{stage}_epoch_{epoch}_class_{class_idx}_weight_map.png'
#     filename_overlay = f'stage_{stage}_epoch_{epoch}_class_{class_idx}_overlay.png'
    
#     # 파일 경로 설정
#     filepath_mask = os.path.join(save_dir, filename_mask)
#     filepath_weight_map = os.path.join(save_dir, filename_weight_map)
#     filepath_overlay = os.path.join(save_dir, filename_overlay)
    
#     # 이미지 저장
#     mask_image.save(filepath_mask)
#     weight_map_image.save(filepath_weight_map)
#     overlay.save(filepath_overlay)
    
#     print(f'Images saved to {save_dir}')