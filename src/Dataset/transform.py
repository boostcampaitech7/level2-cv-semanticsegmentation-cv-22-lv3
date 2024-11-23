# import albumentations as A
# from typing import Callable
# from omegaconf import OmegaConf

# def get_transforms(augmentations_config) -> Callable:
#     augmentations_config = OmegaConf.to_container(augmentations_config, resolve=True)
    

#     transforms_list = []
#     for aug in augmentations_config:
#         if isinstance(aug, dict):
#             for aug_name, aug_params in aug.items():
#                 try:
#                     aug_class = getattr(A, aug_name)
#                     if aug_params is None:
#                         transforms_list.append(aug_class())


#                     else:
#                         transforms_list.append(aug_class(**aug_params))


#                 except AttributeError:
#                     print(f"Albumentations에 '{aug_name}' 증강이 존재하지 않습니다.")

                    
#         else:
#             try:
#                 aug_class = getattr(A, aug)
#                 transforms_list.append(aug_class())


#             except AttributeError:
#                 print(f"Albumentations에 '{aug}' 증강이 존재하지 않습니다.")
          
                    
#     return A.Compose(transforms_list)


import albumentations as A
from typing import Callable
from omegaconf import OmegaConf

def get_transforms(augmentations_config) -> Callable:
    augmentations_config = OmegaConf.to_container(augmentations_config, resolve=True)
    
    transforms_list = []
    for aug in augmentations_config:
        if isinstance(aug, dict):
            for aug_name, aug_params in aug.items():
                try:
                    if aug_name == 'CustomCrop':
                        # x, y, width, height를 x_min, y_min, x_max, y_max로 변환
                        x = aug_params.get('x', 0)
                        y = aug_params.get('y', 0)
                        width = aug_params.get('width', 0)
                        height = aug_params.get('height', 0)
                        crop = A.Crop(
                            x_min=x,
                            y_min=y,
                            x_max=x + width,
                            y_max=y + height,
                            p=aug_params.get('p', 1.0)
                        )
                        transforms_list.append(crop)
                    else:
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