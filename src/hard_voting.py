import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
import pytz
import torch
from utils.inference_utils import encode_mask_to_rle, encode_mask_to_rle_gpu, decode_rle_to_mask, decode_rle_to_mask_gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple CSV submissions using voting")
    parser.add_argument('--csvs', nargs='+', required=True, help='List of CSV files to ensemble')
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help="Ensemble mode: 'cpu' or 'gpu'")
    args = parser.parse_args()

    device = torch.device('cuda') if args.mode == 'gpu' else torch.device('cpu')

    csv_files = args.csvs
    dataframes = []

    # Read all CSV files
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    num_entries = len(dataframes[0])
    for df in dataframes:
        if len(df) != num_entries:
            raise ValueError("CSV files do not have the same number of entries.")

    # Ensure all CSV files have matching 'image_name' and 'class' columns
    for idx in range(num_entries):
        image_name = dataframes[0].iloc[idx]['image_name']
        class_name = dataframes[0].iloc[idx]['class']
        for df in dataframes[1:]:
            if df.iloc[idx]['image_name'] != image_name or df.iloc[idx]['class'] != class_name:
                raise ValueError(f"Mismatch in image_name or class at index {idx}")

    final_image_names = []
    final_classes = []
    final_rles = []

    # Perform voting-based ensemble
    for idx in tqdm(range(num_entries), desc='Ensembling'):
        image_name = dataframes[0].iloc[idx]['image_name']
        class_name = dataframes[0].iloc[idx]['class']
        rle_list = [df.iloc[idx]['rle'] for df in dataframes]

        if args.mode == 'cpu':
            masks = [decode_rle_to_mask(rle) for rle in rle_list]
            masks = np.stack(masks, axis=0)
            votes = np.sum(masks, axis=0)
            threshold = len(masks) / 2
            final_mask = (votes > threshold).astype(np.uint8)
            final_rle = encode_mask_to_rle(final_mask)
        else:
            masks = [decode_rle_to_mask_gpu(rle, device) for rle in rle_list]
            masks = torch.stack(masks, dim=0)
            votes = torch.sum(masks, dim=0)
            threshold = len(masks) / 2
            final_mask = (votes > threshold).to(torch.uint8)
            final_rle = encode_mask_to_rle_gpu(final_mask)

        final_image_names.append(image_name)
        final_classes.append(class_name)
        final_rles.append(final_rle)

    # Create the final DataFrame and save to CSV
    final_df = pd.DataFrame({
        'image_name': final_image_names,
        'class': final_classes,
        'rle': final_rles
    })

    # KST 시간대 설정
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

    # 결과 파일 저장 경로 설정 (project/results 폴더)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project 디렉토리 경로
    output_folder = os.path.join(base_dir, 'results')  # results 폴더 경로
    os.makedirs(output_folder, exist_ok=True)  # results 폴더가 없으면 생성
    output_filepath = os.path.join(output_folder, f'{timestamp}_hard_voting_ensembled.csv')

    # DataFrame을 CSV로 저장
    final_df.to_csv(output_filepath, index=False)
    print(f"Ensemble submission saved to: {output_filepath}")
