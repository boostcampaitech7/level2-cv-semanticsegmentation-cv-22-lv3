from omegaconf import OmegaConf
import argparse
from train.trainer import set_seed
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.Dataset.dataloader import get_test_loader
from model.model_loader import model_loader
import os
from datetime import datetime
import pytz

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def save_to_csv(filename_and_class, rles, cfg):
    # 출력 디렉토리가 없으면 생성
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    
    # 현재 시간을 파일명에 추가
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    pt_name = cfg.inference.checkpoint_path.split('/')[-1].split('.')[0]
    output_filepath = os.path.join(cfg.inference.output_dir, f'{pt_name}_{timestamp}.csv')
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    # DataFrame 생성 및 저장
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    # CSV 파일로 저장
    df.to_csv(output_filepath, index=False)
    print(f"Submission file saved to: {output_filepath}")
    
    return output_filepath

def do_inference(cfg):
    set_seed(cfg.seed)

    CLASS2IND = {v: i for i, v in enumerate(cfg.data.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    # 모델 초기화
    model = model_loader(cfg)

    # config에서 checkpoint path 가져오기
    checkpoint_path = cfg.inference.checkpoint_path
    
    try:
        # 체크포인트 로딩 방식 수정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 체크포인트 타입에 따른 처리
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # 모델이 직접 저장된 경우
            model = checkpoint
            
        print(f"Checkpoint loaded successfully from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # GPU로 모델 이동
    model = model.to(device)
    
    try:
        model.eval()
        rles = []
        filename_and_class = []

        # output_size를 튜플로 변환
        output_size = tuple(map(int, cfg.inference.output_size))

        with torch.no_grad():
            test_loader = get_test_loader(cfg)
            for images, image_names in tqdm(test_loader, total=len(test_loader), desc="Inference"):
                images = images.to(device)
                outputs = model(images)
                
                # outputs가 dictionary인 경우 'out' 키로 접근
                if isinstance(outputs, dict):
                    outputs = outputs['out']

                # interpolate 함수 호출 시 align_corners 파라미터 추가
                outputs = F.interpolate(outputs, 
                                     size=output_size, 
                                     mode='bilinear')
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > cfg.inference.threshold).detach().cpu().numpy()

                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                        
        # CSV 파일로 저장
        output_path = save_to_csv(filename_and_class, rles, cfg)
        print(f"Total predictions: {len(rles)}")

        return output_path
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Semantic Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the experiment config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fcn_resnet50_best_model.pt', help='Path to the pretrained model pt file')

    args = parser.parse_args()

    # cfg 파일 로드
    cfg = OmegaConf.load(args.config)
    cfg.inference.checkpoint_path = args.checkpoint
    
    # inference 실행
    output_path = do_inference(cfg)
    print(f"Inference completed. Results saved to: {output_path}")

