import wandb
import argparse
from omegaconf import OmegaConf
from utils.set_seed import set_seed
from utils.config_utils import ConfigManager
from model.model_loader import model_loader
from Dataset.dataloader import get_train_val_loader
from train.trainer import train
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader


def do_train(cfg: OmegaConf, project_name: str, run_name: str) -> None:
    '''
    summary: `cfg`에 제공된 설정을 사용해 모델을 학습하고, 
              Weights and Biases(wandb)에 학습 관련 지표를 기록하며, 
              모델을 평가하는 함수입니다.

    args: 
        cfg (OmegaConf): 모델, 데이터, 학습 파라미터를 포함한 설정 객체.
        project_name (str): wandb 프로젝트 이름.
        run_name (str): wandb에서 특정 학습 실행을 식별하기 위한 고유 이름.
    
    return: 
        None: 이 함수는 값을 반환하지 않습니다.
    '''

    if cfg.debug:
        cfg.data.train.max_epoch = 2
        cfg.data.train.print_step = 1
        cfg.data.valid.interval = 1

    model = model_loader(cfg)
    criterion = loss_func_loader(cfg)
    optimizer = optimizer_loader(cfg, model.parameters())

    wandb.init(
        project=project_name,  
        name=run_name,         
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    wandb.watch(model, log = 'all')

    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loader(cfg)
    scheduler = lr_scheduler_loader(cfg, optimizer)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg)

    wandb.finish()

    return


# 이 스크립트는 명령줄 인자로 제공된 설정으로 semantic segentation 모델을 학습하는 데 사용됩니다.
# wandb를 활용해 학습 과정과 지표를 로깅하며, 사용자 정의 모델, 인코더, 학습 설정을 지원합니다.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Semantic Segmentation Model')
    parser.add_argument('--config', type=str, default='../configs/base_config.yaml', 
                        help='Path to the config file for train')
    parser.add_argument('--model', type=str, default='../configs/fcn_resnet50.yaml', 
                        help='Path to the model config file')
    parser.add_argument('--encoder', type=str, default=None, 
                        help='Path to the encoder config file')
    parser.add_argument('--project_name', type = str, default='이름 미지정 프로젝트',
                        help='Write a wandb project name')
    parser.add_argument('--run_name', type=str, default='이름 미지정 실험',
                        help='Write a wandb run name')
    args = parser.parse_args()

    config_manager = ConfigManager(
        base_config=args.config, 
        model_config=args.model, 
        encoder_config=args.encoder
        )
    
    config = config_manager.load_config()

    do_train(config, args.project_name, args.run_name)