# test_data_load.py
# 데이터가 잘 로드되고 있는지 확인 용도

import os
from dataset import get_file_lists

image_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/Image_data/train/DCM'
label_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-22-lv3/Image_data/train/outputs_json'

try:
    pngs, jsons = get_file_lists(image_root, label_root)
    print(f"Number of training images: {len(pngs)}")
    print(f"Number of training labels: {len(jsons)}")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")