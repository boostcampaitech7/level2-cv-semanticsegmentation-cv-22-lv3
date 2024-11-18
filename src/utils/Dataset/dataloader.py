
import yaml
import argparse
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from utils.Dataset.dataset import XRayDataset
from utils.Dataset.transform import get_transforms



def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    return config


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
        batch_size = config.data.train.batch_size,
        shuffle = config.data.train.shuffle,
        num_workers = config.data.train.num_workers,
        pin_memory = config.data.train.pin_memory,
        drop_last = config.data.train.drop_last
    )


    val_loader = DataLoader(
        dataset = val_datasets,
        batch_size = config.data.valid.batch_size,
        shuffle = config.data.valid.shuffle,
        num_workers = config.data.valid.num_workers,
        pin_memory = config.data.valid.pin_memory,
        drop_last = config.data.valid.drop_last
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
        batch_size = config.data.test.batch_size,
        shuffle = config.data.test.shuffle,
        num_workers = config.data.test.num_workers,
        pin_memory = config.data.test.pin_memory,
        drop_last = config.data.test.drop_last
    )

    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Path to the config file')
    args = parser.parse_args()
    config = load_config(args.config)


    train_loader, val_loader = get_train_val_loader(config)
    test_loader = get_test_loader(config)


    print("Checking train loader...")
    for images, labels in train_loader:
        print(f"Train batch - images shape: {images.shape}, labels shape: {labels.shape}")
        break  


    print("Checking validation loader...")
    for images, labels in val_loader:
        print(f"Validation batch - images shape: {images.shape}, labels shape: {labels.shape}")
        break  

    # Test loader 확인
    print("Checking test loader...")
    for images, image_names in test_loader:
        print(f"Test batch - images shape: {images.shape}, image names: {image_names}")
        break  


    print("\nDataset counts:")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")


    train_images = set(train_loader.dataset.imagenames)
    val_images = set(val_loader.dataset.imagenames)
    overlap = train_images.intersection(val_images)
    if not overlap:
        print("Train and Validation datasets do not overlap.")
    else:
        print(f"Overlap between Train and Validation datasets: {overlap}")


    print(f"Total test images loaded: {len(test_loader.dataset)}")