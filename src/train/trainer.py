import os
import datetime
import pytz
import pytz
import torch
import wandb
from utils.set_seed import set_seed
from validation.validation import validation
from model.utils.model_output import get_model_output


def save_model(model, file_name='fcn_resnet50_best_model.pt', config=None):
    save_ckpt = config.save.save_ckpt
    save_path = os.path.join(save_ckpt, file_name)
    torch.save(model.state_dict(), save_path)


def train(model, train_loader, val_loader, criterion, optimizer, config) -> None:
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
    print(f'max_epoch: {config.data.train.max_epoch},valid & save_interval: {config.data.valid.interval}')
    print(f'Start training..')

    set_seed(config.seed)
    kst = pytz.timezone('Asia/Seoul')

    best_dice = 0.0
    
    epochs_no_improve = 0
    patience = config.data.train.early_stopping_patience 
    delta = config.data.train.early_stopping_delta  


    os.makedirs(config.save.save_ckpt, exist_ok=True)


    for stage in range(1, 3):
        if stage == 1 :
            stage_epoch = int(config.data.train.max_epoch * config.data.train.ratio)
            stage_trainloader = train_loader
            stage_valloader = val_loader
        else:
            stage_epoch = config.data.train.max_epoch - int(config.data.train.max_epoch * config.data.train.ratio) 
            stage_trainloader = val_loader
            stage_valloader = train_loader


        print(f'Stage_{stage}...')
        print(f'The number of train dataset : {len(train_loader)}')
        print(f'The number of train dataset : {len(val_loader)}')


        for epoch in range(stage_epoch):
            model.train()

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
                        f'Stage{stage} || Epoch [{epoch+1}/{int(config.data.train.max_epoch * config.data.train.ratio)}] | '
                        f'Step [{step+1}/{len(stage_trainloader)}] | '
                        f'Loss: {round(loss.item(),4)}'
                    )
                    wandb.log({'Stage{stage} : train_loss' : loss.item(), 'Epoch' : epoch + 1 })
                

            if (epoch + 1) % config.data.valid.interval == 0:
                dice = validation(epoch + 1, model, stage_valloader, criterion, config=config)


                save_model(model, file_name=f'epoch_{epoch+1}_model.pt', config=config)
                print(f"Save epoch {epoch+1}model in {config.save.save_ckpt}")


                if best_dice < dice + delta:
                    print(f"Stage{stage} Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    print(f"Stage{stage} Save best model in {config.save.save_ckpt}")
                    best_dice = dice
                    epochs_no_improve = 0


                    save_model(model, file_name=f'{config.model.architecture.base_model}_best_model.pt', config=config)
                    best_model_filename = f'{config.model.architecture.base_model}_best_model.pt'
                    best_model_path = os.path.join(config.save.save_ckpt, best_model_filename)
                    best_artifact = wandb.Artifact('best-model', type='model')
                    best_artifact.add_file(best_model_path)
                    wandb.log_artifact(best_artifact)


                else:
                    epochs_no_improve += 1
                    print(f'No improvement in Dice Coefficient for {epochs_no_improve} epochs')
                    print(f"Stage{stage} performance at epoch: {epoch + 1}, {dice:.4f}, dice score decline : {(best_dice - dice):.4f}")


                    if epochs_no_improve >= patience :
                        print(f"Stage{stage} Early stopping triggered after {patience} epochs with no improvement.")
                        print(f"Stage{stage} Best Dice Coefficient: {best_dice:.4f} at epoch {(epoch + 1) - epochs_no_improve}")
                        continue
        
