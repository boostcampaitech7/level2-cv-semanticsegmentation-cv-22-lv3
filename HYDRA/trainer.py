# trainer.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import dice_coef, save_model
import os

try:
    import wandb
except ImportError:
    wandb = None

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    모델 학습 및 검증, 체크포인트 관리, Early Stopping, 로깅 등 학습 전반을 담당합니다.
    """
    def __init__(self, model, model_name, criterion, optimizer, scheduler, device, config, threshold=0.5, phase="phase1"):
        self.model = model
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.threshold = threshold
        self.best_dice = 0.0
        self.writer = None
        self.phase = phase

        self.early_stopping_enabled = config.early_stopping.enabled
        self.early_stopping_patience = config.early_stopping.patience
        self.early_stopping_mode = config.early_stopping.mode
        self.early_stopping_min_delta = config.early_stopping.min_delta
        self.early_stopping_counter = 0
        self.early_stopping_best = self.best_dice if self.early_stopping_mode == 'max' else float('inf')
        self.top_models = []

        if self.config.LOG_TOOL == 'wandb' and wandb:
            # Construct run_name based on config
            run_name = f"{model_name}_{self.config.model.encoder_name}_{self.config.scheduler.name}_{self.config.optimizer.name}_{self.config.loss.name}_{self.config.NUM_EPOCHS_PHASE1}"
            wandb.init(project='xray-segmentation_JH_HZ', config=dict(self.config), name=run_name)
            self.use_wandb = True
        elif self.config.LOG_TOOL == 'tensorboard':
            self.writer = SummaryWriter(log_dir=self.config.SAVED_DIR)
            self.use_wandb = False
        else:
            print("No supported logging tool found. Proceeding without logging.")
            self.use_wandb = False

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        global_step = epoch * len(train_loader)
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images = images.to(self.device)
            masks = masks.to(self.device)

            if self.model_name == 'segformer':
                outputs = self.model(pixel_values=images).logits
            else:
                model_output = self.model(images)
                outputs = model_output['out'] if isinstance(model_output, dict) and 'out' in model_output else model_output

            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if self.use_wandb:
                wandb.log({'train_loss': loss.item()})
            if self.writer:
                current_step = global_step + batch_idx
                self.writer.add_scalar('Loss/train', loss.item(), current_step)

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader, epoch):
        self.model.eval()
        dices = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.model_name == 'segformer':
                    outputs = self.model(pixel_values=images).logits
                else:
                    model_output = self.model(images)
                    outputs = model_output['out'] if isinstance(model_output, dict) and 'out' in model_output else model_output

                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > self.threshold).float()
                dice = dice_coef(outputs, masks)
                dices.append(dice)

        avg_dice = torch.mean(torch.cat(dices, 0)).item()

        if self.use_wandb:
            wandb.log({'val_dice': avg_dice})
        if self.writer:
            self.writer.add_scalar('Dice/val', avg_dice, epoch)

        return avg_dice

    def train(self, train_loader, val_loader, num_epochs):
        """
        주어진 epoch 수만큼 학습을 진행하고, VAL_EVERY 마다 검증을 수행합니다.
        Early Stopping 및 상위 모델 관리도 수행합니다.
        """
        try:
            for epoch in range(num_epochs):
                print(f"Starting Epoch {epoch+1}/{num_epochs}")
                avg_loss = self.train_epoch(train_loader, epoch)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

                if (epoch + 1) % self.config.VAL_EVERY == 0:
                    val_dice = self.validate(val_loader, epoch)
                    print(f"Validation Dice Coefficient: {val_dice:.4f}")
                    model_filename = f"{self.phase}_epoch_{epoch+1}_dice_{val_dice:.4f}.pt"
                    save_model(self.model, self.config.SAVED_DIR, model_filename)
                    print(f"Saved model {model_filename}")

                    self.top_models.append((val_dice, model_filename))
                    self.top_models.sort(key=lambda x: x[0], reverse=True)
                    if len(self.top_models) > 3:
                        for val_dice_removed, model_filename_removed in self.top_models[3:]:
                            try:
                                os.remove(os.path.join(self.config.SAVED_DIR, model_filename_removed))
                                print(f"Removed model {model_filename_removed} with dice score {val_dice_removed:.4f}")
                            except OSError as e:
                                print(f"Error deleting model file {model_filename_removed}: {e}")
                        self.top_models = self.top_models[:3]

                    if val_dice > self.best_dice:
                        self.best_dice = val_dice

                    # Early Stopping
                    self.handle_early_stopping(val_dice)
                    # Scheduler step
                    if self.scheduler:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_dice)
                        else:
                            self.scheduler.step()

                # If VAL_EVERY == 0, scheduler steps on training loss
                if self.config.VAL_EVERY == 0 and self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(avg_loss)
                    else:
                        self.scheduler.step()
        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise
        finally:
            if self.writer:
                self.writer.close()
            if self.use_wandb:
                wandb.finish()

    def handle_early_stopping(self, val_dice):
        """
        Early Stopping 로직을 처리하는 헬퍼 함수.
        """
        improvement = False
        if self.early_stopping_mode == 'max':
            if val_dice > self.early_stopping_best + self.early_stopping_min_delta:
                improvement = True
        elif self.early_stopping_mode == 'min':
            if val_dice < self.early_stopping_best - self.early_stopping_min_delta:
                improvement = True
        else:
            raise ValueError(f"Invalid early_stopping mode: {self.early_stopping_mode}")

        if improvement:
            self.early_stopping_best = val_dice
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            print(f"Early Stopping Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            if self.early_stopping_counter >= self.early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                raise StopIteration("EarlyStopping")
