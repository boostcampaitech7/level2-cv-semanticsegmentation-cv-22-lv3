import os
import cv2
import json
import yaml
import torch
import argparse
import numpy as np
import albumentations as A
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset
from Dataset.transform import get_transforms
from Dataset.split_dataset.splitdata import split_data


def load_config(config_path: str):
    '''
        summary : config파일을 로드
        args : config 파일
        retun : OmegaConf로 로드한 config 파일
    '''
    config = OmegaConf.load(config_path)
    return config


def check_image_label_pair(config) -> tuple[list[str], list[str]]:
    '''
        summary : 이미지이름과 라벨이름이 동일하고 갯수가 맞는지 확인한다
        args : config 파일
        retun : 이미지리스트와 라벨 리스트
    '''
    train_data_path = config.data.train_data_path
    train_label_path = config.data.train_label_path

    images = {
        os.path.relpath(os.path.join(root, fname), start = train_data_path)
        for root, _, files in os.walk(train_data_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.png'
    }

    labels = {
        os.path.relpath(os.path.join(root, fname), start = train_label_path)
        for root, _, files in os.walk(train_label_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.json'
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in labels}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in images}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    images = sorted(images)
    labels = sorted(labels)


    return images, labels


def load_test_images(config):
    '''
        summary : 테스트 데이터를 로드한다
        args : config 파일
        retun : 테스트 데이터 이미지, 테스트 데이터 경로
    '''     
    test_data_path = config.data.test_data_path
    images = sorted([
        os.path.relpath(os.path.join(root, fname), start=test_data_path)
        for root, _, files in os.walk(test_data_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.png'
    ])


    return images, test_data_path


class XRayDataset(Dataset):
    '''
        summary : XRay 이미지에 대한 사용자 커스텀 데이터 클래스
    '''
    def __init__ (self, mode='train', transforms=None, config=None):
        '''
            summary : 필요한 파라미터들을 정의
            args : mode 설정, 증강기법, config 파일
            retun : None
        '''
        self.mode = mode
        self.transforms = transforms
        self.config = config

        if mode in ['train', 'val']:
            images, labels = check_image_label_pair(config)
            _imagenames = np.array(images)
            _labelnames = np.array(labels)

            left_right_group = [os.path.dirname(fname) for fname in _imagenames]
            imagenames, labelnames = split_data(
                _imagenames, _labelnames, left_right_group,
                config=config, mode=mode,
                split_method=config.data.get('split_method', 'GroupKFold')
            )

            if config.debug == False:
                self.imagenames = imagenames
                self.labelnames = labelnames
            elif config.debug == True:
                self.imagenames = imagenames[:30]
                self.labelnames = labelnames[:30]

        elif mode == 'test':
            self.imagenames, self.test_data_path = self.load_test_images(config)
            self.labelnames = []  

        else:
            raise ValueError("Invalid mode. Choose 'train', 'val', or 'test'.")


        self.boundary_width = config.loss_func.get('boundary_width', 5)
        self.weight_inside = config.loss_func.get('weight_inside', 1.0)
        self.weight_boundary = config.loss_func.get('weight_boundary', 2.0)


    def load_test_images(self, config: Dict[str, Any]) -> Tuple[list, str]:
        '''
            summary : 테스트 데이터를 로드
            args : config  파일
            retun : 테스트 이미지, 데이트 데이터 경로
        '''
        test_data_path = config.data.test_data_path
        images = sorted([
            os.path.relpath(os.path.join(root, fname), start=test_data_path)
            for root, _, files in os.walk(test_data_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == '.png'
        ])
        return images, test_data_path


    def create_weight_map(self, mask: np.ndarray) -> np.ndarray:
        '''
            summary : 클래스별 가중치 맵 생성
            args : mask 정보
            retun : 클래스별 가중치 맵들
        '''
        mask = mask.astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        

        boundary = mask.copy()
        for _ in range(self.boundary_width):
            eroded = cv2.erode(boundary, kernel, iterations=1)
            boundary = boundary - eroded
        

        weight_map = np.ones_like(mask, dtype=np.float32) * self.weight_inside
        weight_map[boundary == 1] = self.weight_boundary
        return weight_map


    def __len__(self) -> int:
        '''
            summary : 이미지의 갯수를 반환
            args : None
            retun : 이미지 길이
        '''
        return len(self.imagenames)

    def __getitem__(self, idx: int) -> tuple:
        '''
            summary : 원하는 인덱스에 접근 가능하도록 설정
            args : 인덱스 값
            retun : test인 경우 -> 선택된 이미지, 선택된 이미지 이름
                    train 경우 :
                        가중치 맵 사용 : 이미지, 이미지라벨, 가중치맵
                        가중치 맵 사용X : 이미지, 이미지 라벨
        '''
        image_name = self.imagenames[idx]

        if self.mode == 'test':
            image_path = os.path.join(self.config.data.test_data_path, image_name)
            image = cv2.imread(image_path)
            image = image / 255.0

            if self.transforms is not None:
                inputs = {'image': image}
                result = self.transforms(**inputs)
                image = result['image']

            image = image.transpose(2, 0, 1)  
            image = torch.from_numpy(image).float()
            return image, image_name 
        
        else:
            image_path = os.path.join(self.config.data.train_data_path, image_name)
            image = cv2.imread(image_path)
            image = image / 255.0

            label_name = self.labelnames[idx]
            label_path = os.path.join(self.config.data.train_label_path, label_name)
            label_shape = tuple(image.shape[:2]) + (len(self.config.data.classes),)        
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path , 'r') as f:
                annotations = json.load(f)
            annotations = annotations['annotations']
            
            CLASS2IND = {v: i for i , v in enumerate(self.config.data.classes)}
            IND2CLASS = {v: k for k , v in CLASS2IND.items()}

            for ann in annotations:
                c = ann['label']
                class_id = CLASS2IND[c]
                points = np.array(ann['points'])
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
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

            weight_maps = []
            for c in range(label.shape[0]):
                wm = self.create_weight_map(label[c].numpy())
                weight_maps.append(wm)
            weight_maps = np.stack(weight_maps) 
            weight_maps = torch.from_numpy(weight_maps).float() 

            if self.config.loss_func.weight_map == True :
                return image, label, weight_maps
            else:
                return image, label

