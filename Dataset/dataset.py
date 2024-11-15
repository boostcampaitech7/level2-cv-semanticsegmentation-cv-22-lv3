import yaml
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import GroupKFold
import cv2
import json
import torch
import albumentations as A
import argparse
import matplotlib.pyplot as plt
from transform import get_transforms
from split_dataset.splitdata import split_data



def load_config(config_path : str) -> dict:
    with open(config_path, 'r') as data_config :
        data_config = yaml.safe_load(data_config)
    return data_config


def check_image_label_pair(config) -> tuple[list[str], list[str]]:
    train_data_path = config['data']['train_data_path']
    train_label_path = config['data']['train_label_path']

    # config에서 경로를 불러와 .png 파일들을 저장한다
    images = {
        os.path.relpath(os.path.join(root, fname), start = train_data_path)
        for root, _, files in os.walk(train_data_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.png'
    }

    # config에서 셩로를 불러와 .json 파일들을 저장한다
    labels = {
        os.path.relpath(os.path.join(root, fname), start = train_label_path)
        for root, _, files in os.walk(train_label_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.json'
    }

    # 두 파일의 확장자를 제거하여 파일 이름을 저장한다
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in labels}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in images}

    # 두 파일이름의 차이가 존재한다면 서로 매칭되지 않는 라벨과 이미지이므로 , 에러를 발생시킨다.
    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    images = sorted(images)
    labels = sorted(labels)

    return images, labels


def load_test_images(config):
            test_data_path = config['data']['test_data_path']
            images = sorted([
                os.path.relpath(os.path.join(root, fname), start=test_data_path)
                for root, _, files in os.walk(test_data_path)
                for fname in files
                if os.path.splitext(fname)[1].lower() == '.png'
            ])
            return images, test_data_path


class XRayDataset(Dataset):
    def __init__ (self, mode='train', transforms=None, config=None):

        if mode == 'train' or mode == 'val' : 
            images, labels = check_image_label_pair(config)
            _imagenames = np.array(images)
            _labelnames = np.array(labels)

        else:
            images, labels = load_test_images(config)
            _imagenames = np.array(images)
            _labelnames = np.array(images)


        # 한 사람의 왼손, 오른손을 하나의 그룹으로 묶어준다.
        left_right_group = [os.path.dirname(fname) for fname in _imagenames]


        imagenames, labelnames = split_data(
            _imagenames, _labelnames, left_right_group,
            config=config, mode=mode,
            split_method=config['data'].get('split_method', 'GroupKFold')
        )

        



        self.imagenames = imagenames
        self.labelnames = labelnames
        self.mode = mode
        self.transforms = transforms
        self.config = config

    def __len__(self) -> int:
        return len(self.imagenames)
    

    def __getitem__(self, item : int) -> tuple[list[float], list[float]]:
        image_name = self.imagenames[item]

        if self.mode == 'test':
            image_path = os.path.join(self.config['data']['test_data_path'], image_name)
            image = cv2.imread(image_path)
            # 필요한 전처리 적용
            if self.transforms is not None:
                inputs = {'image': image}
                result = self.transforms(**inputs)
                image = result['image']

            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            return image, image_name 


        else:
            image_path = os.path.join(self.config['data']['train_data_path'], image_name)

            image = cv2.imread(image_path)
            # image = image / 255 

            label_name = self.labelnames[item]
            label_path = os.path.join(self.config['data']['train_label_path'], label_name)
            
            # 라벨의 형태를 생성한다 : 이미지의 높이와 넓이 + 클래스 수 의 형태로 만든들고, 모두 0으로 만들어준다 ( H, W, Class)
            label_shape = tuple(image.shape[:2]) + (len(self.config['data']['class']),)        
            label = np.zeros(label_shape, dtype = np.uint8)

            # label의 annotation 정보를 가지고 온다.
            with open(label_path , 'r') as f:
                annotations = json.load(f)
            annotations = annotations['annotations']
            
            CLASS2IND = {v: i for i , v in enumerate(self.config['data']['class'])}
            IND2CLASS = {v: k for k , v in CLASS2IND.items()}

            for ann in annotations:
                c = ann['label']
                class_id = CLASS2IND[c]
                points = np.array(ann['points'])

                # polygone type을 mask type으로 변경하는 코드를 작성합니다.
                class_label = np.zeros(image.shape[:2], dtype = np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_id] = class_label

            
            if self.transforms is not None:
                inputs = {'image': image, 'mask': label}
                result = self.transforms(**inputs)
                image = result['image']
                label = result['mask']
                

        
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            return image, label



if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # 설정 파일 불러오기
    config = load_config(args.config)

    # 증강 설정 불러오기
    train_transforms_config = config.get('augmentation', {}).get('train', [])
    valid_transforms_config = config.get('augmentation', {}).get('valid', [])

    # 증강 파이프라인 생성
    train_transforms = get_transforms(train_transforms_config)
    valid_transforms = get_transforms(valid_transforms_config)

    # 데이터셋 생성
    train_dataset = XRayDataset(mode='train', transforms=train_transforms, config=config)
    valid_dataset = XRayDataset(mode='val', transforms=valid_transforms, config=config)

    # 데이터셋 사용 예시
    image, label = train_dataset[0]
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Train dataset length:", len(train_dataset))


    # 시각화를 위한 팔레트를 설정합니다.
    PALETTE = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    ]


    def label2rgb(label) -> list:
        image_size = label.shape[1:] + (3, )
        image = np.zeros(image_size, dtype=np.uint8)
        
        for i, class_label in enumerate(label):
            image[class_label == 1] = PALETTE[i]
            
        return image

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image[0])    
    ax[1].imshow(label2rgb(label))

    plt.show()

    
