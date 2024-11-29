import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
import pytz
import torch
import cv2
from utils.inference_utils import simple_postprocessing, encode_mask_to_rle, decode_rle_to_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply post-processing to a CSV submission")
    parser.add_argument('--csv', type=str, required=True, help='CSV file to process')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for post-processing')
    args = parser.parse_args()

    # CSV 파일 읽기
    df = pd.read_csv(args.csv)

    # 결과 저장용 리스트
    processed_rles = []

    # 각 행에 대해 후처리 적용
    for idx in tqdm(range(len(df)), desc='Post-processing'):
        # RLE 디코딩
        original_mask = decode_rle_to_mask(df.iloc[idx]['rle'])
        
        # 후처리 적용
        processed_mask = simple_postprocessing(
            original_mask, 
            threshold=args.threshold
        )
        
        # RLE 인코딩
        processed_rle = encode_mask_to_rle(processed_mask)
        processed_rles.append(processed_rle)

    # 새로운 DataFrame 생성
    processed_df = df.copy()
    processed_df['rle'] = processed_rles

    # KST 시간대 설정
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

    # 결과 파일 저장 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_folder = os.path.join(base_dir, 'results')
    os.makedirs(output_folder, exist_ok=True)
    
    # 원본 파일명 추출
    original_filename = os.path.splitext(os.path.basename(args.csv))[0]
    output_filepath = os.path.join(output_folder, f'{timestamp}_{original_filename}_postprocessed.csv')

    # DataFrame을 CSV로 저장
    processed_df.to_csv(output_filepath, index=False)
    print(f"Post-processed submission saved to: {output_filepath}")