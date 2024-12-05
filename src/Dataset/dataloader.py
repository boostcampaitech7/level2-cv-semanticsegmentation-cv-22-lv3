
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from Dataset.dataset import XRayDataset
from src.Dataset.utils.transform import get_transforms


def get_train_val_loader(config : Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    '''
        summary : 
            데이터셋을 커스텀 데이터셋과 데이터 로더를 활용하여 커스텀된 train_laoder와 val_loader를 생성합니다.
        args : 
            config 파일
        retun : 
            trainloader와 valloader
    '''
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
    '''
        summary : 
            데이터셋을 커스텀 데이터셋과 데이터 로더를 활용하여 커스텀된 test_loader를 생성합니다.
        args : 
            config 파일
        retun : 
            Test Data Loader
    '''
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

