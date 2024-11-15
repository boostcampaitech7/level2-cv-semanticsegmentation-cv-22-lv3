from typing import Callable
import albumentations as A



def get_transforms(augmentations_config : dict) -> Callable:
    transforms_list = []
    for aug in augmentations_config:
        if isinstance(aug, dict):
            for aug_name, aug_params in aug.items():
                try:
                    # Albumentations에서 해당 증강 클래스 가져오기
                    aug_class = getattr(A, aug_name)
                    if aug_params is None:
                        transforms_list.append(aug_class())
                    else:
                        transforms_list.append(aug_class(**aug_params))
                except AttributeError:
                    print(f"Albumentations에 '{aug_name}' 증강이 존재하지 않습니다.")
        else:
            # 파라미터 없이 증강 이름만 지정한 경우
            try:
                aug_class = getattr(A, aug)
                transforms_list.append(aug_class())
            except AttributeError:
                print(f"Albumentations에 '{aug}' 증강이 존재하지 않습니다.")
    return A.Compose(transforms_list)