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
    config = OmegaConf.load(config_path)
    return config


def check_image_label_pair(config) -> tuple[list[str], list[str]]:
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
            test_data_path = config.data.test_data_path
            images = sorted([
                os.path.relpath(os.path.join(root, fname), start=test_data_path)
                for root, _, files in os.walk(test_data_path)
                for fname in files
                if os.path.splitext(fname)[1].lower() == '.png'
            ])


            return images, test_data_path


class XRayDataset(Dataset):
    def __init__ (self, mode='train', transforms=None, config=None):
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

            if config.debug == False :
                self.imagenames = imagenames
                self.labelnames = labelnames


            elif config.debug == True :
                self.imagenames = images[ : 30]
                self.labelnames = labelnames [ : 30]


        elif mode == 'test':
            self.imagenames, self.test_data_path = self.load_test_images(config)
            self.labelnames = []  


        else:
            raise ValueError("Invalid mode. Choose 'train', 'val', or 'test'.")


    def load_test_images(self, config: Dict[str, Any]) -> Tuple[list, str]:
        test_data_path = config.data.test_data_path
        images = sorted([
            os.path.relpath(os.path.join(root, fname), start=test_data_path)
            for root, _, files in os.walk(test_data_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == '.png'
        ])


        return images, test_data_path


    def __len__(self) -> int:

        return len(self.imagenames)
    

    def __getitem__(self, item : int) -> tuple[list[float], list[float]]:
        image_name = self.imagenames[item]

        if self.mode == 'test':
            image_path = os.path.join(self.config.data.test_data_path, image_name)
            image = cv2.imread(image_path)
            image = image / 255.


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
            image = image / 255 


            label_name = self.labelnames[item]
            label_path = os.path.join(self.config.data.train_label_path, label_name)
            

            label_shape = tuple(image.shape[:2]) + (len(self.config.data.classes),)        
            label = np.zeros(label_shape, dtype = np.uint8)


            with open(label_path , 'r') as f:
                annotations = json.load(f)
            annotations = annotations['annotations']
            
            CLASS2IND = {v: i for i , v in enumerate(self.config.data.classes)}
            IND2CLASS = {v: k for k , v in CLASS2IND.items()}

            for ann in annotations:
                c = ann['label']
                class_id = CLASS2IND[c]
                points = np.array(ann['points'])


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
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Path to the config file')
    args = parser.parse_args()
    config = load_config(args.config)


    train_transforms_config = config.get('augmentation', {}).get('train', [])
    valid_transforms_config = config.get('augmentation', {}).get('valid', [])


    train_transforms = get_transforms(train_transforms_config)
    valid_transforms = get_transforms(valid_transforms_config)


    train_dataset = XRayDataset(mode='train', transforms=train_transforms, config=config)
    valid_dataset = XRayDataset(mode='val', transforms=valid_transforms, config=config)


    image, label = train_dataset[0]
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Train dataset length:", len(train_dataset))


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

    