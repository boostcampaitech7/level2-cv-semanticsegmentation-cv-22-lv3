# dataset.py

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, train_test_split

class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, image_root, label_root, config, is_train=True, transforms=None, preprocessing=None, feature_extractor=None, normalize=True):
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
        self.normalize = normalize  # normalize 인자 저장

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            # 이미지 로드
            image_name = self.filenames[idx]
            image_path = os.path.join(self.image_root, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")

            # BGR에서 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 라벨 로드
            label_name = self.labelnames[idx]
            label_path = os.path.join(self.label_root, label_name)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {label_path}")

            # 라벨 생성
            label_shape = image.shape[:2] + (len(self.classes), )
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]

            for ann in annotations:
                class_name = ann["label"]
                if class_name not in self.class2ind:
                    raise ValueError(f"알 수 없는 클래스 이름: {class_name}")
                class_idx = self.class2ind[class_name]
                points = np.array(ann["points"], dtype=np.int32)
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_idx] = class_label

            # normalize 플래그에 따라 이미지 정규화 적용
            if self.normalize:
                image = image / 255.0

            # Apply transforms
            if self.transforms:
                inputs = {"image": image, "mask": label}
                outputs = self.transforms(**inputs)
                image = outputs["image"]
                label = outputs.get("mask", label)

            # Apply preprocessing
            if self.preprocessing:
                image = self.preprocessing(image)

            # Feature extractor 적용
            if self.feature_extractor:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
                label = torch.from_numpy(label.transpose(2, 0, 1)).long()
            else:
                # 채널 차원 변경 및 Tensor 변환
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                label = torch.from_numpy(label.transpose(2, 0, 1)).float()

            return image, label
        except Exception as e:
            print(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
            raise

def get_file_lists(image_root, label_root, image_ext='.png', label_ext='.json'):
    # 이미지 파일 리스트 얻기
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files
        if fname.lower().endswith(image_ext)
    }

    # 라벨 파일 리스트 얻기
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _, files in os.walk(label_root)
        for fname in files
        if fname.lower().endswith(label_ext)
    }

    # 파일 이름에서 확장자를 제거하여 비교
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}

    # 이미지와 라벨 파일 수가 일치하는지 확인
    if pngs_fn_prefix != jsons_fn_prefix:
        missing_in_images = jsons_fn_prefix - pngs_fn_prefix
        missing_in_labels = pngs_fn_prefix - jsons_fn_prefix
        raise ValueError(f"이미지와 라벨 파일이 일치하지 않습니다.\n"
                         f"이미지에 없는 라벨 파일: {missing_in_images}\n"
                         f"라벨에 없는 이미지 파일: {missing_in_labels}")

    # 정렬된 리스트 반환
    return sorted(pngs), sorted(jsons)

def split_dataset(filenames, labelnames, mode='kfold', n_splits=5, val_fold=0, random_seed=2024):
    if len(filenames) == 0 or len(labelnames) == 0:
        raise ValueError("파일 리스트가 비어 있습니다. 경로를 확인하세요.")

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
        # 80:20으로 랜덤 분할
        train_filenames, val_filenames, train_labelnames, val_labelnames = train_test_split(
            filenames, labelnames, test_size=0.2, random_state=random_seed, shuffle=True
        )

        return train_filenames, train_labelnames, val_filenames, val_labelnames

    else:
        raise ValueError(f"Invalid data_split_mode: {mode}")

def get_test_file_list(image_root, image_ext='.png'):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files
        if fname.lower().endswith(image_ext)
    }
    return sorted(pngs)

class XRayInferenceDataset(Dataset):
    def __init__(self, filenames, image_root, transforms=None, preprocessing=None, feature_extractor=None, normalize=True):
        self.filenames = filenames
        self.image_root = image_root
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor
        self.normalize = normalize  # normalize 인자 저장

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            image_name = self.filenames[idx]
            image_path = os.path.join(self.image_root, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")

            # BGR에서 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.normalize:
                image = image / 255.0

            if self.transforms:
                image = self.transforms(image=image)["image"]
            
            # Preprocessing 함수 호출 수정
            if self.preprocessing:
                image = self.preprocessing(image)
                # preprocessing이 함수이므로 직접 호출

            # Feature extractor 적용
            if self.feature_extractor:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            return image, image_name
        except Exception as e:
            print(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
            raise
