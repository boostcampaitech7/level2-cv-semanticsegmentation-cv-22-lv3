import wandb
import argparse
from omegaconf import OmegaConf
from train.trainer import train
from utils.set_seed import set_seed
from configs.utils import ConfigManager
from model.model_loader import model_loader
from utils.Dataset.dataloader import get_train_val_loader
from train.loss_opt_sche import loss_func_loader, lr_scheduler_loader, optimizer_loader


def do_train(cfg, project_name, run_name):
    if cfg.debug:
        cfg.train.max_epoch = 2
        cfg.train.print_step = 1
        cfg.val.interval = 1


    model, _ = model_loader(cfg)
    criterion = loss_func_loader(cfg)
    optimizer = optimizer_loader(cfg, model.parameters())


    wandb.init(
        project=project_name,  
        name=run_name,         
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    wandb.watch(model, log = 'all')


    # Scheduler 선택
    # scheduler = lr_scheduler_loader(cfg, optimizer)


    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loader(cfg)
    train(model, train_loader, val_loader, criterion, optimizer, cfg)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")

    parser.add_argument('--config', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/configs/base_config.yaml', 
                        help='Path to the config file for train')
    

    parser.add_argument('--model', type=str, 
                        default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/src/model/torchvision/configs/fcn_resnet50.yaml', 
                        help='Path to the model config file')
    

    parser.add_argument('--encoder', type=str, 
                        default=None, 
                        help='Path to the encoder config file')
    

    parser.add_argument('--project_name', type = str,
                        default='이름 미지정 프로젝트',
                        help='Write a wandb project name'
                        )
    

    parser.add_argument('--run_name', type=str,
                        default='이름 미지정 실험',
                        help='Write a wandb run name'
                        )

    args = parser.parse_args()
    

    config_manager = ConfigManager(base_config=args.config,
                         model_config=args.model,
                         encoder_config=args.encoder,
                         )
    

    config = config_manager.load_config()
    do_train(config, args.project_name, args.run_name)
