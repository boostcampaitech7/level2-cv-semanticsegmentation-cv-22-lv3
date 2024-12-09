import os
import os.path as osp
import pytz
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from inference_utils import postprocessing_with_sharpening, encode_mask_to_rle, decode_rle_to_mask


# 샤프닝 필터 적용 및 모폴로지 연산을 통해 예측 이미지의 가장자리를 뚜렷하게 합니다.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply post-processing to a CSV submission')
    parser.add_argument('--csv', type=str, required=True, help='CSV file to process')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for post-processing')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    processed_rles = []

    for idx in tqdm(range(len(df)), desc='Post-processing'):
        original_mask = decode_rle_to_mask(df.iloc[idx]['rle'])
        processed_mask = postprocessing_with_sharpening(
            original_mask, 
            threshold=args.threshold
        )
        
        processed_rle = encode_mask_to_rle(processed_mask)
        processed_rles.append(processed_rle)

    processed_df = df.copy()
    processed_df['rle'] = processed_rles

    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')

    base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    output_folder = osp.join(base_dir, 'results')
    os.makedirs(output_folder, exist_ok=True)
    
    original_filename = osp.splitext(osp.basename(args.csv))[0]
    output_filepath = osp.join(output_folder, f'{timestamp}_{original_filename}_postprocessed.csv')

    processed_df.to_csv(output_filepath, index=False)
    print(f'Post-processed submission saved to: {output_filepath}')
    