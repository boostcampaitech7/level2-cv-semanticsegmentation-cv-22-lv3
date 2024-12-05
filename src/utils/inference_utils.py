import os
import cv2
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import torch
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
from set_seed import set_seed


def prepare_inference_environment(configs : list) -> tuple:
    '''
    summary :
        추론을 위한 환경을 준비하는 함수로, 시드 설정, 클래스 매핑, 디바이스 설정을 수행합니다.

    args : 
        configs : config 파일 경로 리스트

    return :
        tuple(device, 클래스-인덱스 매핑, 인덱스-클래스 매핑)
    '''
    set_seed(configs.seed)
    
    CLASS2IND = {v: i for i, v in enumerate(configs[0].data.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    device = torch.device('cuda') if configs[0].inference.mode == 'gpu' else torch.device('cpu')
    
    return device, CLASS2IND, IND2CLASS


def encode_mask_to_rle(mask: np.ndarray) -> str:
    '''
    summary :
        마스크를 RLE(Run Length Encoding) 문자열로 인코딩합니다.

    args : 
        mask : 인코딩할 마스크 배열

    return :
        RLE 인코딩된 문자열
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def encode_mask_to_rle_gpu(mask: torch.Tensor) -> str:
    '''
    summary :
        GPU에서 마스크를 RLE 문자열로 인코딩합니다.

    args : 
        mask : 인코딩할 마스크 텐서

    return :
        RLE 인코딩된 문자열
    '''
    mask = mask.to(torch.uint8)
    pixels = mask.flatten()
    zeros = torch.zeros(1, dtype=pixels.dtype, device=pixels.device)
    pixels = torch.cat([zeros, pixels, zeros])
    changes = (pixels[1:] != pixels[:-1]).nonzero().squeeze() + 1
    runs = changes.clone()
    runs[1::2] = runs[1::2] - runs[::2]
    runs = runs.cpu().numpy()
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle : str, height : int = 2048, width : int = 2048) -> np.ndarray:
    '''
    summary :
        RLE 문자열을 마스크로 디코딩합니다.

    args : 
        rle : RLE 인코딩된 문자열
        height : 마스크의 높이. 기본값은 2048.
        width : 마스크의 너비. 기본값은 2048.

    return :
        numpy.ndarray: 디코딩된 마스크 배열
    '''
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


def decode_rle_to_mask_gpu(rle : str, device : torch.device, height : int = 2048, width : int = 2048) -> torch.Tensor:
    '''
    summary :
        GPU에서 RLE(Run Length Encoding) 문자열을 마스크로 디코딩합니다.

    args : 
        rle : RLE 인코딩된 문자열
        device : 텐서를 로드할 디바이스
        height : 마스크의 높이. 기본값은 2048.
        width : 마스크의 너비. 기본값은 2048.

    return :
        디코딩된 마스크 텐서
    '''
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


def postprocessing_with_sharpening(mask : np.ndarray, threshold : float = 0.5) -> np.ndarray:
    '''
    summary :
        OpenCV 기반의 샤프닝 및 후처리 함수로, 마스크의 품질을 향상시킵니다.

    args : 
        mask : 입력 마스크 배열
        threshold : 이진화 임계값. 기본값은 0.5.

    return :
        후처리된 마스크 배열
    '''
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    
    sharpened_mask = cv2.filter2D(mask.astype(np.float32), -1, sharpening_kernel)
    binary_mask = (sharpened_mask > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    return opened_mask


def save_to_csv(filename_and_class : list, rles : list, cfg : OmegaConf):
    '''
    summary :
        추론 결과를 CSV 파일로 저장합니다.

    args : 
        filename_and_class : 파일명과 클래스 정보의 리스트
        rles : RLE 인코딩된 마스크 문자열 리스트
        cfg : 설정 객체

    return :
        저장된 CSV 파일의 경로
    '''
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    pt_name = cfg.inference.checkpoint_path.split('/')[-1].split('.')[0]
    output_filepath = os.path.join(cfg.inference.output_dir, f'{pt_name}_{timestamp}.csv')

    classes, filename = zip(*[x.split('_') for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        'image_name': image_name,
        'class': classes,
        'rle': rles,
    })

    df.to_csv(output_filepath, index=False)
    print(f'Submission file saved to: {output_filepath}')
    return output_filepath


def load_model(model : torch.nn.Module, checkpoint_path : str, device : torch.device, library : str) -> torch.nn.Module:
    '''
    summary :
        사전 훈련된 모델 체크포인트를 로드하고 지정된 디바이스에 배치합니다.

    args : 
        model : 초기화된 모델 객체
        checkpoint_path : 체크포인트 파일 경로
        device : 모델을 로드할 장치
        library : 사용하는 모델 라이브러리

    return :
        로드된 모델
    '''
    checkpoint = torch.load(checkpoint_path, map_location=device) if library == 'torchvision' else smp.from_pretrained(checkpoint_path)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    print(f'Checkpoint loaded successfully from: {checkpoint_path}')
    return model