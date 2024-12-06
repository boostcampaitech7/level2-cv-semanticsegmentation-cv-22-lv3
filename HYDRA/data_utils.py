# data_utils.py
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_file_lists, split_dataset, XRayDataset

def get_preprocessing(preprocessing_fn=None, feature_extractor=None):
    """
    이 함수는 전처리 함수를 반환합니다.
    feature_extractor를 사용하는 경우 별도 전처리가 필요하지 않고,
    preprocessing_fn이 지정되면 해당 함수를 반환합니다.
    명시되지 않은 경우 None을 반환합니다.
    """
    if feature_extractor:
        return None
    elif preprocessing_fn:
        return preprocessing_fn
    else:
        return None

def get_transforms(config, is_train=True):
    """
    이 함수는 config에 정의된 augmentation 설정을 바탕으로
    Albumentations Compose 객체를 생성하여 반환합니다.
    """
    if is_train:
        augmentations = config.augmentations.train
    else:
        augmentations = config.augmentations.val

    transform_list = []

    print("Applying the following augmentations:")
    for aug in augmentations:
        aug_type = aug['type']
        params = aug.get('params', {})
        if hasattr(A, aug_type):
            aug_class = getattr(A, aug_type)
            transform_instance = aug_class(**params)
            transform_list.append(transform_instance)
            print(f"- {aug_type} with params {params}")
        else:
            raise ValueError(f"Augmentation {aug_type} not found in albumentations")

    return A.Compose(transform_list)

def prepare_data(config):
    """
    이 함수는 데이터셋 파일 목록을 가져오고,
    config에 설정된 split 모드(kfold, two_phase 등)에 따라
    훈련/검증 데이터를 분리하여 반환합니다.
    """
    pngs, jsons = get_file_lists(config.IMAGE_ROOT, config.LABEL_ROOT)
    print(f"Number of training images: {len(pngs)}")
    print(f"Number of training labels: {len(jsons)}")

    if len(pngs) == 0 or len(jsons) == 0:
        raise ValueError("No training images or labels found. Check the paths.")

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

def create_datasets(train_files, train_labels, val_files, val_labels, config, preprocessing_fn, feature_extractor=None, normalize=True):
    """
    이 함수는 주어진 파일 목록, 라벨 목록, config 등을 바탕으로
    훈련 및 검증용 Dataset 객체(XRayDataset)를 생성하여 반환합니다.
    """
    from data_utils import get_transforms, get_preprocessing  # 자체 모듈 내 재활용
    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    train_preproc = get_preprocessing(preprocessing_fn, feature_extractor)
    val_preproc = get_preprocessing(preprocessing_fn, feature_extractor)

    train_dataset = XRayDataset(
        filenames=train_files,
        labelnames=train_labels,
        image_root=config.IMAGE_ROOT,
        label_root=config.LABEL_ROOT,
        config=config,
        is_train=True,
        transforms=train_transforms,
        preprocessing=train_preproc,
        feature_extractor=feature_extractor,
        normalize=normalize
    )

    valid_dataset = XRayDataset(
        filenames=val_files,
        labelnames=val_labels,
        image_root=config.IMAGE_ROOT,
        label_root=config.LABEL_ROOT,
        config=config,
        is_train=False,
        transforms=val_transforms,
        preprocessing=val_preproc,
        feature_extractor=feature_extractor,
        normalize=normalize
    )
    return train_dataset, valid_dataset

def create_dataloaders(train_dataset, valid_dataset, config):
    """
    이 함수는 Dataset 객체로부터 DataLoader를 생성하여 반환합니다.
    """
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
