# data_utils.py

import numpy as np
from dataset import get_file_lists, split_dataset, XRayDataset
from torch.utils.data import DataLoader
import albumentations as A

def get_preprocessing(preprocessing_fn=None, feature_extractor=None):
    if feature_extractor:
        return None  # feature_extractor를 사용하는 경우 별도의 전처리가 필요 없음
    elif preprocessing_fn:
        return preprocessing_fn  # preprocessing_fn을 직접 반환
    else:
        return None

def get_transforms(config, is_train=True):
    # Build augmentations based on config
    if is_train:
        augmentations = config.augmentations.train
    else:
        augmentations = config.augmentations.val

    # List to store the transformations
    transform_list = []

    print("Applying the following augmentations:")
    for aug in augmentations:
        aug_type = aug['type']
        params = aug.get('params', {})
        # Get the augmentation class from albumentations
        if hasattr(A, aug_type):
            aug_class = getattr(A, aug_type)
            transform_instance = aug_class(**params)
            transform_list.append(transform_instance)
            print(f"- {aug_type} with params {params}")
        else:
            raise ValueError(f"Augmentation {aug_type} is not found in albumentations")

    # Create the Compose object
    return A.Compose(transform_list)

def prepare_data(config):
    # 데이터 준비
    pngs, jsons = get_file_lists(config.IMAGE_ROOT, config.LABEL_ROOT)
    print(f"Number of training images: {len(pngs)}")
    print(f"Number of training labels: {len(jsons)}")

    if len(pngs) == 0 or len(jsons) == 0:
        raise ValueError("학습 이미지 또는 라벨 파일이 비어 있습니다. 경로를 확인하세요.")
 
    train_filenames, train_labelnames, val_filenames, val_labelnames = split_dataset(
        filenames=np.array(pngs),
        labelnames=np.array(jsons),
        mode=config.get("data_split_mode", "kfold"),
        n_splits=5,
        val_fold=0,
        random_seed=config.RANDOM_SEED
    )
    print(f"Training set: {len(train_filenames)}, Validation set: {len(val_filenames)}")

    return train_filenames, train_labelnames, val_filenames, val_labelnames

def create_datasets(train_filenames, train_labelnames, val_filenames, val_labelnames, config, preprocessing_fn, feature_extractor=None, normalize=True):
    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    # Get preprocessing transformations
    train_preprocessing = get_preprocessing(preprocessing_fn, feature_extractor)
    val_preprocessing = get_preprocessing(preprocessing_fn, feature_extractor)

    train_dataset = XRayDataset(
        filenames=train_filenames,
        labelnames=train_labelnames,
        image_root=config.IMAGE_ROOT,
        label_root=config.LABEL_ROOT,
        config=config,
        is_train=True,
        transforms=train_transforms,
        preprocessing=train_preprocessing,
        feature_extractor=feature_extractor,
        normalize=normalize  # normalize 인자 추가
    )

    valid_dataset = XRayDataset(
        filenames=val_filenames,
        labelnames=val_labelnames,
        image_root=config.IMAGE_ROOT,
        label_root=config.LABEL_ROOT,
        config=config,
        is_train=False,
        transforms=val_transforms,
        preprocessing=val_preprocessing,
        feature_extractor=feature_extractor,
        normalize=normalize  # normalize 인자 추가
    )
    return train_dataset, valid_dataset


def create_dataloaders(train_dataset, valid_dataset, config):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.get("num_workers", 8)
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("num_workers", 8)
    )

    return train_loader, valid_loader
