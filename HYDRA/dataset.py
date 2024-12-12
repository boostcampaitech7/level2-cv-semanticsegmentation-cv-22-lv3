# dataset.py
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, train_test_split

class XRayDataset(Dataset):
    """
    X-ray 이미지와 라벨(마스크) 데이터셋.
    이미지 읽기, 라벨 생성, augmentation 및 preprocessing 등을 수행.
    """
    def __init__(self, filenames, labelnames, image_root, label_root, config,
                 is_train=True, transforms=None, preprocessing=None, feature_extractor=None, normalize=True):
        self.filenames = filenames
        self.labelnames = labelnames
        self.image_root = image_root
        self.label_root = label_root
        self.classes = config.CLASSES
        self.class2ind = config.CLASS2IND
        self.is_train = is_train
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor
        self.normalize = normalize

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        주어진 인덱스의 이미지/라벨을 로드하고,
        transforms 및 preprocessing을 적용한 후 Tensor를 반환합니다.
        """
        try:
            image_name = self.filenames[idx]
            image_path = os.path.join(self.image_root, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label_name = self.labelnames[idx]
            label_path = os.path.join(self.label_root, label_name)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Cannot find label file: {label_path}")

            label_shape = image.shape[:2] + (len(self.classes), )
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]

            for ann in annotations:
                class_name = ann["label"]
                if class_name not in self.class2ind:
                    raise ValueError(f"Unknown class name: {class_name}")
                class_idx = self.class2ind[class_name]
                points = np.array(ann["points"], dtype=np.int32)
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_idx] = class_label

            if self.normalize:
                image = image / 255.0

            if self.transforms:
                outputs = self.transforms(image=image, mask=label)
                image = outputs["image"]
                label = outputs["mask"]

            if self.preprocessing:
                image = self.preprocessing(image)

            if self.feature_extractor:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
                label = torch.from_numpy(label.transpose(2, 0, 1)).long()
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                label = torch.from_numpy(label.transpose(2, 0, 1)).float()

            return image, label
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            raise

def get_file_lists(image_root, label_root, image_ext='.png', label_ext='.json'):
    """
    이미지/라벨 파일 리스트를 얻고, 일치하는지 확인합니다.
    """
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files if fname.lower().endswith(image_ext)
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _, files in os.walk(label_root)
        for fname in files if fname.lower().endswith(label_ext)
    }

    png_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    json_prefix = {os.path.splitext(fname)[0] for fname in jsons}

    if png_prefix != json_prefix:
        missing_in_images = json_prefix - png_prefix
        missing_in_labels = png_prefix - json_prefix
        raise ValueError(f"Files do not match.\nMissing in images: {missing_in_images}\nMissing in labels: {missing_in_labels}")

    return sorted(pngs), sorted(jsons)

def split_dataset(filenames, labelnames, mode='kfold', n_splits=5, val_fold=0, random_seed=2024):
    """
    주어진 파일 리스트를 kfold 또는 two_phase 모드로 분할합니다.
    """
    if len(filenames) == 0 or len(labelnames) == 0:
        raise ValueError("Empty file list.")

    if mode == 'kfold':
        groups = [os.path.dirname(fname) for fname in filenames]
        ys = [0] * len(filenames)
        gkf = GroupKFold(n_splits=n_splits)

        train_filenames, train_labelnames = [], []
        val_filenames, val_labelnames = [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(filenames, ys, groups)):
            if fold == val_fold:
                val_filenames = [filenames[i] for i in val_idx]
                val_labelnames = [labelnames[i] for i in val_idx]
            else:
                train_filenames.extend([filenames[i] for i in train_idx])
                train_labelnames.extend([labelnames[i] for i in train_idx])

        return train_filenames, train_labelnames, val_filenames, val_labelnames

    elif mode == 'two_phase':
        train_files, val_files, train_lbls, val_lbls = train_test_split(
            filenames, labelnames, test_size=0.2, random_state=random_seed, shuffle=True
        )
        return train_files, train_lbls, val_files, val_lbls
    else:
        raise ValueError(f"Invalid mode: {mode}")

def get_test_file_list(image_root, image_ext='.png'):
    """
    테스트 이미지 파일 리스트를 반환합니다.
    """
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files if fname.lower().endswith(image_ext)
    }
    return sorted(pngs)

class XRayInferenceDataset(Dataset):
    """
    추론용 Dataset. 라벨 없이 이미지만 로드하여 전처리/특징 추출 후 반환.
    """
    def __init__(self, filenames, image_root, transforms=None, preprocessing=None, feature_extractor=None, normalize=True):
        self.filenames = filenames
        self.image_root = image_root
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor
        self.normalize = normalize

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            image_name = self.filenames[idx]
            image_path = os.path.join(self.image_root, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.normalize:
                image = image / 255.0

            if self.transforms:
                image = self.transforms(image=image)["image"]

            if self.preprocessing:
                image = self.preprocessing(image)

            if self.feature_extractor:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()

            return image, image_name
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            raise
