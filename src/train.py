from torchvision import models
import torch.nn as nn
import torch.optim as optim
from train.trainer import set_seed, train
from utils.Dataset.dataloader import get_train_val_loader
from omegaconf import OmegaConf
import argparse
import torch
import numpy as np


def do_train(cfg):
    if cfg.debug:
        cfg.train.max_epoch = 2
        cfg.val.interval = 1

    model = models.segmentation.fcn_resnet50(pretrained=True)

    model.classifier[4] = nn.Conv2d(512, len(cfg.data.classes), kernel_size=1)

    # Loss function 선택
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 선택
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    # Scheduler 선택
    

    # 시드를 설정합니다.
    set_seed(cfg.seed)
    
    train_loader, val_loader = get_train_val_loader(cfg)

    # train(model, train_loader, val_loader, criterion, optimizer)
    train(model, train_loader, val_loader, criterion, optimizer, cfg)



if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    # parser.add_argument('--config_train', type=str, default='configs/train/base_train.yaml', help='Path to the train config file')
    parser.add_argument('--config', type=str, default='../configs/base_config.yaml', help='Path to the data config file')

    args = parser.parse_args()

    # 설정 파일 로드
    # config_train = OmegaConf.load(args.config_train)
    config = OmegaConf.load(args.config)

    # do_train(config_train, config)
    do_train(config)