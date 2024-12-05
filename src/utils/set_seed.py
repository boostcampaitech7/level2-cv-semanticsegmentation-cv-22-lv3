import torch
import random
import numpy as np


def set_seed(seed : int) -> None:
    '''
    summary :
        재현성을 위해 모든 랜덤 시드를 고정합니다. PyTorch, CUDA, NumPy, Python의 랜덤 생성기를 초기화합니다.

    args : 
        seed : 고정할 랜덤 시드 값

    return :
        반환값이 없습니다.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    return