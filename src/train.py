from train.trainer import set_seed, train
from utils.Dataset.dataloader import get_train_val_loader
import argparse
from model.model_loader import model_loader
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader
import wandb
from configs.utils import ConfigManager
from omegaconf import OmegaConf

def do_train(cfg):
    if cfg.debug:
        cfg.train.max_epoch = 2
        cfg.train.print_step = 1
        cfg.val.interval = 1

    # model_loader에서 모델과 모델 이름 2개 return 하도록 수정
    model, model_name = model_loader(cfg)

    # Loss function 선택
    criterion = loss_func_loader(cfg)

    # Optimizer 선택
    optimizer = optimizer_loader(cfg, model.parameters())


    # wandb 초기화
    wandb.init(
        project = 'Segmentation Baseline',
        name = f'Run_Name_model_{str(model_name)}',
        config = OmegaConf.to_container(cfg, resolve=True),
        reinit = True
    )
    wandb.watch(model, log = 'all')


    # Scheduler 선택
    # scheduler = lr_scheduler_loader(cfg, optimizer)


    # 시드를 설정합니다.
    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loader(cfg)


    train(model, train_loader, val_loader, criterion, optimizer, cfg)

    wandb.finish()



if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    # parser.add_argument('--config_train', type=str, default='configs/train/base_train.yaml', help='Path to the train config file')
    parser.add_argument('--config', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/base_config.yaml', 
                        help='Path to the config file for train')
    parser.add_argument('--model', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/torchvision/configs/fcn_resnet50.yaml', 
                        help='Path to the model config file')
    parser.add_argument('--encoder', type=str, 
                        default=None, 
                        help='Path to the encoder config file')
    # basemodel_시간 -> 폴더 생성

    args = parser.parse_args()
    
    config_manager = ConfigManager(base_config=args.config,
                         model_config=args.model,
                         encoder_config=args.encoder)
    
    config = config_manager.load_config()

    # do_train(config_train, config)
    do_train(config)