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
    def __init__(self, model, model_name, criterion, optimizer, scheduler, device, config, threshold=0.5, phase="phase1"):
        self.model = model
        self.model_name = model_name  # 모델 이름 저장
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.threshold = threshold
        self.best_dice = 0.0
        self.writer = None  # Initialize self.writer as None
        self.phase = phase  # Phase identifier to differentiate checkpoints

        # Early Stopping 설정
        self.early_stopping_enabled = config.early_stopping.enabled
        self.early_stopping_patience = config.early_stopping.patience
        self.early_stopping_mode = config.early_stopping.mode
        self.early_stopping_min_delta = config.early_stopping.min_delta
        self.early_stopping_counter = 0
        self.early_stopping_best = self.best_dice if self.early_stopping_mode == 'max' else float('inf')

        # 상위 3개의 모델을 저장하기 위한 리스트 초기화
        self.top_models = []  # 리스트 형태로 (dice_score, model_filename) 저장

        # Logging 도구 설정
        if self.config.LOG_TOOL == 'wandb' and wandb:
            # Run 이름 생성: model_name + encoder_name + scheduler + optimizer + loss + epoch수 + augmentations
            model_name_cfg = self.config.model.model_name
            encoder_name = self.config.model.encoder_name
            scheduler_name = self.config.scheduler.name
            optimizer_name = self.config.optimizer.name
            loss_name = self.config.loss.name  # <-- 손실 함수 이름 추가
            num_epochs = self.config.NUM_EPOCHS_PHASE1  # Use Phase1 epochs

            # 증강 방식 추출 및 문자열 변환
            if hasattr(self.config, 'augmentations'):
                aug = self.config.augmentations
                # 각 단계(train, val)의 증강 타입을 추출하여 연결
                train_augs = "_".join([aug_step['type'] for aug_step in aug.get('train', [])])
                val_augs = "_".join([aug_step['type'] for aug_step in aug.get('val', [])])
                # train과 val의 증강을 구분하여 하나의 문자열로 합침
                augmentations_str = f"train-{train_augs}_val-{val_augs}"
            else:
                augmentations_str = "NoAugmentation"
                print("Warning: Augmentation settings not found. Using 'NoAugmentation'.")

            # 수정된 run_name 형식
            run_name = f"{model_name_cfg}_{encoder_name}_{scheduler_name}_{optimizer_name}_{loss_name}_{num_epochs}epoch_aug_{augmentations_str}"
            
            # 디버깅을 위해 run_name 출력
            print(f"Run Name: {run_name}")
            
            wandb.init(
                project='xray-segmentation_JH_HZ',
                config=dict(self.config),
                name=run_name  # Set the created run name
            )
            self.use_wandb = True
        elif self.config.LOG_TOOL == 'tensorboard':
            self.writer = SummaryWriter(log_dir=self.config.SAVED_DIR)
            self.use_wandb = False
        else:
            print("지원하지 않는 로그 도구이거나 wandb가 설치되지 않은 것입니다. 로그 없이 진행합니다.")
            self.use_wandb = False
            # self.writer remains None

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        global_step = epoch * len(train_loader)
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.model_name == 'segformer':
                outputs = self.model(pixel_values=images)
                outputs = outputs.logits
            else:
                model_output = self.model(images)
                if isinstance(model_output, dict) and 'out' in model_output:
                    outputs = model_output['out']
                else:
                    outputs = model_output

            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            # Logging
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
                    outputs = self.model(pixel_values=images)
                    outputs = outputs.logits
                else:
                    model_output = self.model(images)
                    if isinstance(model_output, dict) and 'out' in model_output:
                        outputs = model_output['out']
                    else:
                        outputs = model_output

                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > self.threshold).float()
                dice = dice_coef(outputs, masks)
                dices.append(dice)

        avg_dice = torch.mean(torch.cat(dices, 0)).item()

        # Logging
        if self.use_wandb:
            wandb.log({'val_dice': avg_dice})
        if self.writer:
            self.writer.add_scalar('Dice/val', avg_dice, epoch)

        return avg_dice

    def train(self, train_loader, val_loader, num_epochs):
        try:
            for epoch in range(num_epochs):
                print(f"Starting Epoch {epoch+1}/{num_epochs}")
                avg_loss = self.train_epoch(train_loader, epoch)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

                # Validation
                if (epoch + 1) % self.config.VAL_EVERY == 0:
                    val_dice = self.validate(val_loader, epoch)
                    print(f"Validation Dice Coefficient: {val_dice:.4f}")
                    
                    # 모델 저장 및 상위 3개 모델 관리
                    model_filename = f"{self.phase}_epoch_{epoch+1}_dice_{val_dice:.4f}.pt"
                    save_model(self.model, self.config.SAVED_DIR, model_filename)
                    print(f"Saved model {model_filename}")

                    # top_models 리스트 업데이트
                    self.top_models.append((val_dice, model_filename))
                    # dice score를 기준으로 내림차순 정렬
                    self.top_models.sort(key=lambda x: x[0], reverse=True)
                    # 상위 3개만 유지하고 나머지는 삭제
                    if len(self.top_models) > 3:
                        for val_dice_removed, model_filename_removed in self.top_models[3:]:
                            try:
                                os.remove(os.path.join(self.config.SAVED_DIR, model_filename_removed))
                                print(f"Removed model {model_filename_removed} with dice score {val_dice_removed:.4f}")
                            except OSError as e:
                                print(f"Error deleting model file {model_filename_removed}: {e}")
                        self.top_models = self.top_models[:3]  # 상위 3개만 유지

                    # best_dice 업데이트
                    if val_dice > self.best_dice:
                        self.best_dice = val_dice

                    # Early Stopping Logic
                    if self.early_stopping_enabled:
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
                            self.early_stopping_best = val_dice if self.early_stopping_mode == 'max' else val_dice
                            self.early_stopping_counter = 0
                        else:
                            self.early_stopping_counter += 1
                            print(f"Early Stopping Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                            if self.early_stopping_counter >= self.early_stopping_patience:
                                print("Early stopping triggered. Stopping training.")
                                break

                        # Scheduler step
                        if self.scheduler:
                            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.scheduler.step(val_dice)
                            else:
                                self.scheduler.step()

                # Handle cases where VAL_EVERY does not align with num_epochs
                if self.config.VAL_EVERY == 0:
                    if self.scheduler:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(avg_loss)
                        else:
                            self.scheduler.step()

        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise
        finally:
            # Resource cleanup
            if self.writer:
                self.writer.close()
            if self.use_wandb:
                wandb.finish()
