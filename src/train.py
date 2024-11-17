from torchvision import models
import torch.nn as nn
import torch.optim as optim
from train.trainer import set_seed, train
from utils.Dataset.dataloader import get_train_val_loader
from omegaconf import OmegaConf
import argparse
import torch
import numpy as np
from model.model_loader import model_loader
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader
from train.trainer import load_config

def do_train(cfg):
    if cfg.debug:
        cfg.train.max_epoch = 2
        cfg.val.interval = 1

    model = model_loader(cfg)

    # Loss function 선택
    criterion = loss_func_loader(cfg.loss_name)

    # Optimizer 선택
    optimizer = optimizer_loader(cfg, model.parameters())

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
    parser.add_argument('--config', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/base_config.yaml', 
                        help='Path to the config file for train')
    parser.add_argument('--model', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/torchvision/configs/lr_aspp_mobilenetv3_large.yaml', 
                        help='Path to the model config file')
    parser.add_argument('--encoder', type=str, 
                        default=None, 
                        help='Path to the encoder config file')
    parser.add_argument('--save_dir', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/checkpoints/basemodel', 
                        help='Path to the model save_dir')
    parser.add_argument('--save_config', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/exp_config.yaml', 
                        help='Path to the config file')

    args = parser.parse_args()

    # 설정 파일 로드
    # config_train = OmegaConf.load(args.config_train)
    config = load_config(base_config=args.config,
                         model_config=args.model,
                         encoder_config=args.encoder,
                         save_path=args.save)
    # do_train(config_train, config)
    do_train(config)