import torch
import segmentation_models_pytorch as smp


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