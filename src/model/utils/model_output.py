def get_model_output(model, input):
    outputs = model(input)['out'] if isinstance(model(input), dict) else model(input)
    return outputs


import os
import torch
from omegaconf import OmegaConf
def save_model(model: torch.nn.Module, 
               file_name: str = 'fcn_resnet50_best_model', 
               config: OmegaConf = None) -> None:
    '''
    summary:
        주어진 모델을 지정된 파일 이름과 경로에 저장합니다. 
        모델의 라이브러리에 따라 저장 방식(torchvision 또는 smp)이 결정됩니다.

    args:
        model (torch.nn.Module): 저장할 모델 객체.
        file_name (str): 저장될 파일 이름. 기본값은 'fcn_resnet50_best_model'.
        config (OmegaConf): 모델 저장 설정을 포함하는 구성 객체.
    '''

    library = config.model.library
    file_name = f'{file_name}{".pt" if library == "torchvision" else ""}'
    save_ckpt = config.save.save_ckpt
    save_path = os.path.join(save_ckpt, file_name)

    if library == 'torchvision':
        torch.save(model.state_dict(), save_path)
    else:
        model.save_pretrained(save_path)
