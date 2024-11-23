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
    # smp 모델일 경우
    else:
        model.save_pretrained(save_path)
        

# def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config) -> None:
#     '''
#         summary : We train the model using a two-stage approach. 
#             No. 1: Learn with train and verify with val
#             No. 2: Learn with val and verify with train

#         args : model
#             train_loader
#             val_loader
#             criterion
#             optimizer
#             config
#     '''
#     print(f'max_epoch: {config.data.train.max_epoch},valid & save_interval: {config.data.valid.interval}')
#     print(f'Start training..')

#     set_seed(config.seed)
#     kst = pytz.timezone('Asia/Seoul')

#     best_dice = 0.0
    
#     epochs_no_improve = 0
#     patience = config.data.train.early_stopping_patience 
#     delta = config.data.train.early_stopping_delta  


#     os.makedirs(config.save.save_ckpt, exist_ok=True)


#     for stage in range(1, 3):
#         if stage == 1 :
#             stage_epoch = int(config.data.train.max_epoch)
#             stage_trainloader = train_loader
#             stage_valloader = val_loader
#         else:
#             stage_epoch = config.data.train.max_epoch - int(config.data.train.max_epoch * config.data.train.ratio) 
#             stage_trainloader = val_loader
#             stage_valloader = train_loader


#         print(f'Stage_{stage}...')
#         print(f'The number of train dataset : {len(train_loader)}')
#         print(f'The number of train dataset : {len(val_loader)}')


#         for epoch in range(stage_epoch):
#             model.train()

#             for step, (images, masks) in enumerate(stage_trainloader):            
#                 images, masks = images.cuda(), masks.cuda()
#                 model = model.cuda()
                

#                 outputs = get_model_output(model, config.model.library, images)


#                 loss = criterion(outputs, masks)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                

#                 if (step + 1) % config.data.train.print_step == 0:
#                     print(
#                         f'{datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")} || '
#                         f'Stage{stage} || Epoch [{epoch+1}/{stage_epoch}] | '
#                         f'Step [{step+1}/{len(stage_trainloader)}] | '
#                         f'Loss: {round(loss.item(),4)}'
#                     )
#                     wandb.log({f'Stage{stage} : train_loss' : loss.item(), 'Epoch' : epoch + 1 })
            
#             scheduler.step()   


#             if (epoch + 1) % config.data.valid.interval == 0:
#                 dice = validation(epoch + 1, model, stage_valloader, criterion, config=config)


#                 save_model(model, file_name=f'epoch_{epoch+1}_model', config=config)
#                 print(f"Save epoch {epoch+1}model in {config.save.save_ckpt}")


#                 if best_dice + delta <= dice:
#                     print(f"Stage{stage} Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
#                     print(f"Stage{stage} Save best model in {config.save.save_ckpt}")
#                     best_dice = dice
#                     # epochs_no_improve = 0


#                     save_model(model, file_name=f'{config.model.architecture.base_model}_best_model', config=config)
#                     # best_model_filename = f'{config.model.architecture.base_model}_best_model.pt'
#                     # best_model_path = os.path.join(config.save.save_ckpt, best_model_filename)
#                     # best_artifact = wandb.Artifact('best-model', type='model')
#                     # best_artifact.add_file(best_model_path)
#                     # wandb.log_artifact(best_artifact)


#                 else:
#                     epochs_no_improve += 1
#                     print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')
#                     print(f"Stage{stage} performance at epoch: {epoch + 1}, {dice:.4f}, dice score decline : {(best_dice - dice):.4f}")


#                     if epochs_no_improve >= patience :
#                         print(f"Stage{stage} Early stopping triggered after {patience} epochs with no improvement.")
#                         print(f"Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}")
#                         continue
        




import matplotlib.pyplot as plt
from train.loss_opt_sche import create_weight_map



def visualize_weight_map(mask, weight_map, class_idx, epoch, stage):
    '''
        Args:
            mask (numpy.ndarray): 이진 마스크 (H, W)
            weight_map (numpy.ndarray): 가중치 맵 (H, W)
            class_idx (int): 클래스 인덱스
            epoch (int): 현재 에포크
            stage (int): 현재 스테이지
    '''
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Weight Map")
    plt.imshow(weight_map, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(mask, cmap='gray')
    plt.imshow(weight_map, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()

    # 이미지를 WandB에 로그
    wandb.log({f'Stage{stage} - Epoch{epoch} - Class{class_idx} Weight Map': plt})

    # 플롯을 닫아 메모리 누수 방지
    plt.close()




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
        print(f'The number of train dataset : {len(stage_trainloader)}')
        print(f'The number of val dataset : {len(stage_valloader)}')

        for epoch in range(stage_epoch):
            model.train()
            epoch_loss = 0.0

            for step, (images, masks) in enumerate(stage_trainloader):            
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()

                outputs = get_model_output(model, config.model.library, images)

                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % config.data.train.print_step == 0:
                    print(
                        f'{datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")} || '
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
                    # epochs_no_improve = 0

                    save_model(model, file_name=f'{config.model.architecture.base_model}_best_model', config=config)
                    # best_model_filename = f'{config.model.architecture.base_model}_best_model.pt'
                    # best_model_path = os.path.join(config.save.save_ckpt, best_model_filename)
                    # best_artifact = wandb.Artifact('best-model', type='model')
                    # best_artifact.add_file(best_model_path)
                    # wandb.log_artifact(best_artifact)

                else:
                    epochs_no_improve += 1
                    print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')
                    print(f"Stage{stage} performance at epoch: {epoch + 1}, {dice:.4f}, dice score decline : {(best_dice - dice):.4f}")

                    if epochs_no_improve >= patience:
                        print(f"Stage{stage} Early stopping triggered after {patience} epochs with no improvement.")
                        print(f"Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}")
                        continue

            # 시각화를 원하는 에포크 또는 조건을 설정
            # 예를 들어, 각 스테이지의 첫 번째 에포크에서 시각화
            if epoch == 0:
                # 일부 샘플 선택 (예: 배치의 첫 번째 샘플)
                sample_images, sample_masks = next(iter(stage_trainloader))
                sample_images, sample_masks = sample_images.cuda(), sample_masks.cuda()
                sample_outputs = get_model_output(model, config.model.library, sample_images)
                sample_outputs = torch.sigmoid(sample_outputs).detach().cpu().numpy()
                sample_masks = sample_masks.detach().cpu().numpy()

                # 클래스 몇 개 선택 (예: 첫 3개 클래스)
                selected_classes = [0, 1, 2]  # 필요에 따라 조정

                for class_idx in selected_classes:
                    mask = sample_masks[0, class_idx, :, :]  # 첫 번째 샘플, 특정 클래스
                    weight_map = create_weight_map(mask)

                    visualize_weight_map(mask, weight_map, class_idx, epoch + 1, stage)