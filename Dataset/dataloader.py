from torch.utils.data import DataLoader
from transform import get_transforms
from dataset import XRayDataset
from typing import Dict, Any, Tuple
import yaml
import argparse

def load_config(config_path : str) -> dict:
    with open(config_path, 'r') as data_config :
        data_config = yaml.safe_load(data_config)
    return data_config


def get_train_val_loader(config : Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:

    train_datasets = XRayDataset(
        mode = 'train',
        transforms = get_transforms(config.get('augmentation', {}).get('train', [])),
        config = config
    )

    val_datasets = XRayDataset(
        mode = 'val',
        transforms = get_transforms(config.get('augmentation', {}).get('valid', [])),
        config = config
    )

    train_loader = DataLoader(
        dataset = train_datasets,
        batch_size = config['data']['train']['batch_size'],
        shuffle = config['data']['train']['suffle'],
        num_workers = config['data']['train']['num_workers'],
        pin_memory = config['data']['train']['pin_memory']
    )


    val_loader = DataLoader(
        dataset = val_datasets,
        batch_size = config['data']['val']['batch_size'],
        shuffle = config['data']['val']['suffle'],
        num_workers = config['data']['val']['num_workers'],
        pin_memory = config['data']['val']['pin_memory']
    )


    return train_loader, val_loader


def get_test_loader(config : Dict[str, Any]) -> DataLoader : 
    test_datasets = XRayDataset(
        mode = 'test',
        transforms = get_transforms(config.get('augmentation', {}).get('test', [])),
        config = config
    )


    test_loader = DataLoader(
        dataset = test_datasets,
        batch_size = config['data']['test']['batch_size'],
        shuffle = config['data']['test']['suffle'],
        num_workers = config['data']['test']['num_workers'],
        pin_memory = config['data']['test']['pin_memory']
    )

    return test_loader

if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # 설정 파일 로드
    config = load_config(args.config)

    # Train and validation loaders
    train_loader, val_loader = get_train_val_loader(config)

    # Test loader
    test_loader = get_test_loader(config)

    # Train loader 확인 ( 첫배치만 확인하고 break )
    print("Checking train loader...")
    for images, labels in train_loader:
        print(f"Train batch - images shape: {images.shape}, labels shape: {labels.shape}")
        break  

    # Validation loader 확인
    print("Checking validation loader...")
    for images, labels in val_loader:
        print(f"Validation batch - images shape: {images.shape}, labels shape: {labels.shape}")
        break  

    # Test loader 확인
    print("Checking test loader...")
    for images, image_names in test_loader:
        print(f"Test batch - images shape: {images.shape}, image names: {image_names}")
        break  