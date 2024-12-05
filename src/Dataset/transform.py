import albumentations as A
from typing import Callable
from omegaconf import OmegaConf

def get_transforms(augmentations_config) -> Callable:
    '''
        summary : config파일의 증강 기법을 albumentation으로 적용
        args : base config 파일
        retun : 설정된 증강 기법들을 포함하는 변환 함수
    '''
    augmentations_config = OmegaConf.to_container(augmentations_config, resolve=True)
    transforms_list = []

    for aug in augmentations_config:
        if isinstance(aug, dict):
            for aug_name, aug_params in aug.items():
                try:
                    aug_class = getattr(A, aug_name)
                    if aug_params is None:
                        transforms_list.append(aug_class())
                    else:
                        transforms_list.append(aug_class(**aug_params))
                except AttributeError:
                    print(f"Albumentations에 '{aug_name}' 증강이 존재하지 않습니다.") 

        else:
            try:
                aug_class = getattr(A, aug)
                transforms_list.append(aug_class())
            except AttributeError:
                print(f"Albumentations에 '{aug}' 증강이 존재하지 않습니다.")
                 
    return A.Compose(transforms_list)

