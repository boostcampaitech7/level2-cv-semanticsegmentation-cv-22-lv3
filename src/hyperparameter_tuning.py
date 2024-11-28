import os
import pytz
import wandb
import optuna
import argparse
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train.trainer import train
from omegaconf import OmegaConf
from utils.set_seed import set_seed
from utils.config_utils import ConfigManager
from model.model_loader import model_loader
from Dataset.dataloader import get_train_val_loader
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader


def objective(trial):
    # 1. Optuna로 변경 가능한 하이퍼파라미터 정의
    # 예시: learning rate, optimizer type, loss function 등
    cfg = ConfigManager(
        base_config='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/base_config.yaml',
        model_config='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/torchvision/configs/fcn_resnet50.yaml',
        encoder_config=None,
    ).load_config()

    # 변경 가능한 하이퍼파라미터
    cfg.optimizer.lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    cfg.optimizer.name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    cfg.model.dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    cfg.train.max_epoch = 10  # 예제에서는 10 epoch으로 제한

    # 2. 데이터 로더 및 모델 준비
    train_loader, val_loader = get_train_val_loader(cfg)

    # 모델, 손실 함수, 옵티마이저, 스케줄러 로드
    model = model_loader(cfg)
    criterion = loss_func_loader(cfg)
    optimizer = optimizer_loader(cfg, model.parameters())
    scheduler = lr_scheduler_loader(cfg, optimizer)

    # 3. 모델 학습
    best_val_accuracy = 0.0
    for epoch in range(cfg.train.max_epoch):
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg)

        # 모델 검증
        val_accuracy = evaluate_model(model, val_loader)  # 정의 필요: 정확도를 계산하는 함수
        trial.report(val_accuracy, epoch)

        # 프루닝 조건
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    return best_val_accuracy

def evaluate_model(model, val_loader):
    """
    모델의 정확도를 계산하는 함수.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(cfg.device), target.to(cfg.device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return correct / total

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()  # 또는 SkoptSampler 등
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials, 1-hour timeout

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
