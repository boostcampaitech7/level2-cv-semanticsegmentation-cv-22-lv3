import os
import pytz
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import segmentation_models_pytorch as smp
import cv2


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def encode_mask_to_rle_gpu(mask):
    mask = mask.to(torch.uint8)
    pixels = mask.flatten()
    zeros = torch.zeros(1, dtype=pixels.dtype, device=pixels.device)
    pixels = torch.cat([zeros, pixels, zeros])
    changes = (pixels[1:] != pixels[:-1]).nonzero().squeeze() + 1
    runs = changes.clone()
    runs[1::2] = runs[1::2] - runs[::2]
    runs = runs.cpu().numpy()
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height=2048, width=2048):
    if not isinstance(rle, str) or len(rle.strip()) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    s = rle.strip().split()
    starts = np.array(s[::2], dtype=int) - 1
    lengths = np.array(s[1::2], dtype=int)
    ends = starts + lengths

    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    img = img.reshape((height, width))
    return img

def decode_rle_to_mask_gpu(rle, device, height=2048, width=2048):
    if not isinstance(rle, str) or len(rle.strip()) == 0:
        return torch.zeros((height, width), dtype=torch.uint8, device=device)

    s = rle.strip().split()
    starts = torch.tensor([int(x) for x in s[::2]], dtype=torch.long, device=device) - 1
    lengths = torch.tensor([int(x) for x in s[1::2]], dtype=torch.long, device=device)
    ends = starts + lengths

    if len(starts) > 0:
        ranges = torch.cat([torch.arange(s, e, device=device) for s, e in zip(starts, ends)])
        mask = torch.zeros(height * width, dtype=torch.uint8, device=device)
        mask[ranges] = 1
    else:
        mask = torch.zeros(height * width, dtype=torch.uint8, device=device)

    mask = mask.reshape((height, width))
    return mask

def simple_postprocessing(mask, threshold=0.5):
    """
    OpenCV 기반 간단한 후처리 함수
    
    :param mask: 입력 마스크 (numpy array)
    :param threshold: 이진화 임계값
    :return: 후처리된 마스크
    """
    # 가우시안 블러
    blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    
    # 모폴로지 연산
    kernel = np.ones((3,3), np.uint8)
    
    # 닫기 연산 (작은 구멍 채우기)
    closed_mask = cv2.morphologyEx(
        (blurred_mask > threshold).astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        kernel
    )
    
    # 열기 연산 (노이즈 제거)
    opened_mask = cv2.morphologyEx(
        closed_mask, 
        cv2.MORPH_OPEN, 
        kernel
    )
    
    return opened_mask

def save_to_csv(filename_and_class, rles, cfg):
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    pt_name = cfg.inference.checkpoint_path.split('/')[-1].split('.')[0]
    output_filepath = os.path.join(cfg.inference.output_dir, f'{pt_name}_{timestamp}.csv')

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })


    df.to_csv(output_filepath, index=False)
    print(f"Submission file saved to: {output_filepath}")
    return output_filepath

def load_model(model, checkpoint_path, device, library):
    checkpoint = torch.load(checkpoint_path, map_location=device) if library == 'torchvision' else smp.from_pretrained(checkpoint_path)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    print(f"Checkpoint loaded successfully from: {checkpoint_path}")
    return model