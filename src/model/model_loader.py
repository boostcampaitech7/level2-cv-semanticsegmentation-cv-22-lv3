import torch
import importlib
from Model.utils.modify_model import update_last_layer
from src.Model.utils import load_model


def model_loader(config):
    '''
        summary : 
            설정(config)에 따라 모델을 로드합니다.
        args : 
            모델 설정이 포함된 구성 객체
        return : 
            로드된 모델 객체
    '''

    model_cfg = config.model
    library = model_cfg.library 
    base_model = model_cfg.base_model
    architecture_params = model_cfg.architecture
    num_classes = len(config.data.classes)
    checkpoint = model_cfg.checkpoint
    model_module = importlib.import_module(model_cfg.module)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hasattr(model_module, base_model):
         target_model = getattr(model_module, base_model)
    else:
         raise ValueError(f"Unknown model {base_model} in {model_module}")
    

    if checkpoint:
        model = load_model(target_model, checkpoint, device, library)             
        print(f'Loaded {library} model: {base_model} with {checkpoint}')
    
    else:
        
        if library == 'torchvision':
            model = target_model(**architecture_params)
            model = update_last_layer(model, num_classes)
            print(f'Loaded torchvision model: {base_model} with {num_classes} classes.')
         
        else:
            model = target_model(**architecture_params)        
            print(f'Loaded SMP model: {base_model} with {num_classes} classes.')    
    
    return model