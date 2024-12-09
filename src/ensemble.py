import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import pytz
from tqdm import tqdm
import torch
from utils.inference_utils import encode_mask_to_rle, encode_mask_to_rle_gpu, decode_rle_to_mask, decode_rle_to_mask_gpu


def validate_csv_files(dataframes : pd.DataFrame) -> bool:
    '''
    CSV 파일들의 유효성을 검증합니다.

    summary:
        입력된 CSV 파일들의 엔트리 수와 이미지 이름, 클래스가 일치하는지 확인합니다.

    args:
        dataframes : 검증할 pandas DataFrame 리스트

    return:
        모든 CSV 파일이 유효하면 True, 그렇지 않으면 ValueError 발생
    '''
    num_entries = len(dataframes[0])
    
    for df in dataframes:
        if len(df) != num_entries:
            raise ValueError('The number of entries in the CSV files does not match.')

    for idx in range(num_entries):
        image_name = dataframes[0].iloc[idx]['image_name']
        class_name = dataframes[0].iloc[idx]['class']
        
        for df in dataframes[1:]:
            if df.iloc[idx]['image_name'] != image_name or df.iloc[idx]['class'] != class_name:
                raise ValueError(f'Image_name or class does not match in the {idx} index.')
    
    return True


def perform_ensemble(dataframes : pd.DataFrame, mode : str ='gpu') -> pd.DataFrame:
    '''
    제공된 CSV 파일들에 대해 Hard Voting 앙상블을 수행합니다.

    summary:
        여러 CSV 파일의 RLE 마스크를 통해 Hard Voting 방식으로 최종 마스크를 생성합니다.

    args:
        dataframes : 앙상블할 pandas DataFrame 리스트
        mode : 연산 모드 ('cpu' 또는 'gpu'), 기본값은 'gpu'

    return:
        앙상블된 최종 DataFrame
    '''
    device = torch.device('cuda') if mode == 'gpu' else torch.device('cpu')

    final_image_names = []
    final_classes = []
    final_rles = []

    num_entries = len(dataframes[0])

    for idx in tqdm(range(num_entries), desc='Performing Ensemble...'):
        image_name = dataframes[0].iloc[idx]['image_name']
        class_name = dataframes[0].iloc[idx]['class']
        rle_list = [df.iloc[idx]['rle'] for df in dataframes]

        if mode == 'cpu':
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

    return pd.DataFrame({
        'image_name': final_image_names,
        'class': final_classes,
        'rle': final_rles
    })


def save_ensemble_result(df : pd.DataFrame, base_dir : str) -> str:
    '''
    앙상블 결과를 CSV 파일로 저장합니다.

    summary:
        입력된 DataFrame을 타임스탬프가 포함된 파일명으로 results 폴더에 저장합니다.

    args:
        df : 저장할 DataFrame
        base_dir : 프로젝트 기본 디렉토리 경로

    return:
        저장된 파일 경로
    '''
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')

    output_folder = osp.join(base_dir, 'results')
    os.makedirs(output_folder, exist_ok=True)
    
    output_filepath = osp.join(output_folder, f'{timestamp}_hard_voting_ensembled.csv')
    df.to_csv(output_filepath, index=False)
    
    return output_filepath


def main():
    '''
    CSV 파일들을 Hard Voting 방식으로 앙상블하는 메인 함수입니다.

    summary:
        커맨드라인 인자를 파싱하고 CSV 파일들을 앙상블하여 결과를 저장합니다.
    '''
    parser = argparse.ArgumentParser(description='Ensemble multiple CSV submissions using voting')
    parser.add_argument('--csvs', nargs='+', required=True, help='List of CSV files to ensemble')
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help='Ensemble mode: "cpu" or "gpu"')
    args = parser.parse_args()

    dataframes = [pd.read_csv(csv_file) for csv_file in args.csvs]
    validate_csv_files(dataframes)
    final_df = perform_ensemble(dataframes, args.mode)
    base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    output_filepath = save_ensemble_result(final_df, base_dir)
    
    print(f'Results saved to: {output_filepath}')


if __name__ == '__main__':
    main()