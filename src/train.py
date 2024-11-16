from torchvision import models
import torch.nn as nn
import torch.optim as optim
from train.trainer import set_seed, train
from utils.Dataset.dataloader import get_train_val_loader
from omegaconf import OmegaConf
import argparse
import torch
import numpy as np
from model.torchvision.model_loader import model_loader

def do_train(cfg):
    if cfg.debug:
        cfg.train.max_epoch = 2
        cfg.val.interval = 1

    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # model.classifier[4] = nn.Conv2d(512, len(cfg.data.classes), kernel_size=1)
    model = model_loader(cfg)

    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    # 시드를 설정합니다.
    set_seed(cfg.seed)
    
    train_loader, val_loader = get_train_val_loader(cfg)

    # train(model, train_loader, val_loader, criterion, optimizer, config_train, config)
    train(model, train_loader, val_loader, criterion, optimizer, config)



if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    # parser.add_argument('--config_train', type=str, default='configs/train/base_train.yaml', help='Path to the train config file')
    parser.add_argument('--config', type=str, default='configs/data/config.yaml', help='Path to the data config file')

    args = parser.parse_args()

    # 설정 파일 로드
    # config_train = OmegaConf.load(args.config_train)
    config = OmegaConf.load(args.config)

    # do_train(config_train, config)
    do_train(config)