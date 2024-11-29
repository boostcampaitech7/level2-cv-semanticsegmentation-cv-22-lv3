import optuna
import argparse
import torch.nn as nn
from train import validation
from omegaconf import OmegaConf
from train.trainer import train
import torch.nn.functional as F
from optuna.trial import TrialState
from model.model_loader import model_loader
from optuna.integration import SkoptSampler
from utils.config_utils import ConfigManager
from Dataset.dataloader import get_train_val_loader
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader


def objective(trial):
    # 1. Optuna로 변경 가능한 하이퍼파라미터 정의
    # 예시: learning rate, optimizer type, loss function 등
    cfg = ConfigManager(
        base_config='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/base_config.yaml',
        model_config='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/smp/configs/base_unet.yaml',
        encoder_config='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/smp/configs/encoder/mobilenet/timm_mobilenetv3_small_minimal_100.yaml',
    ).load_config()

    # 변경 가능한 하이퍼파라미터
    cfg.optimizer.lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    cfg.optimizer.name = trial.suggest_categorical("optimizer", ["AdamW", "SGD", "RMSprop"])
    cfg.data.train.max_epoch = 10  # 예제에서는 10 epoch으로 제한

    # 2. 데이터 로더 및 모델 준비
    train_loader, val_loader = get_train_val_loader(cfg)

    # 모델, 손실 함수, 옵티마이저, 스케줄러 로드
    model = model_loader(cfg)
    criterion = loss_func_loader(cfg)
    optimizer = optimizer_loader(cfg, model.parameters())
    scheduler = lr_scheduler_loader(cfg, optimizer)

    # 3. 모델 학습
    best_val_dice = 0.0
    for epoch in range(cfg.data.train.max_epoch):
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg)

        # 모델 검증
        val_dice = validation(cfg.data.train.max_epoch, model, val_loader, cfg)
        trial.report(val_dice, epoch)

        # 프루닝 조건
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_dice = max(best_val_dice, val_dice)

    return best_val_dice


if __name__ == "__main__":
    sampler = SkoptSampler(
        skopt_kwargs={'n_random_starts': 5,
                      'acq_func': 'EI',
                      'acq_func_kwargs': {'xi': 0.02}}
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials, 1-hour timeout

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
