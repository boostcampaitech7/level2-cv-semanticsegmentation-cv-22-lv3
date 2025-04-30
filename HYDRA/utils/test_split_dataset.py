# test_split_dataset.py

import os
from dataset import get_file_lists, split_dataset
import numpy as np

image_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/Image_data/train/DCM'
label_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/Image_data/train/outputs_json'

try:
    pngs, jsons = get_file_lists(image_root, label_root)
    print(f"Total images: {len(pngs)}, Total labels: {len(jsons)}")
    train_filenames, train_labelnames, val_filenames, val_labelnames = split_dataset(
        filenames=np.array(pngs),
        labelnames=np.array(jsons),
        mode='two_phase',
        n_splits=5,
        val_fold=0,
        random_seed=42
    )
    print(f"Training set: {len(train_filenames)}")
    print(f"Validation set: {len(val_filenames)}")
except Exception as e:
    print(f"데이터 분할 중 오류 발생: {e}")
