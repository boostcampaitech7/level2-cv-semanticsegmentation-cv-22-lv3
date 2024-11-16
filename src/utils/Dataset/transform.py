from typing import Callable
import albumentations as A
from omegaconf import OmegaConf

def get_transforms(augmentations_config) -> Callable:
    # OmegaConf 객체를 일반 파이썬 객체로 변환
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