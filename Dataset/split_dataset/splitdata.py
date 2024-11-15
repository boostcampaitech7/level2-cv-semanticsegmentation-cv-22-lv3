from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import os

def split_data(
    _imagenames: list,
    _labelnames: list,
    groups: list,
    config: dict,
    mode: str = 'train',
    split_method: str = "GroupKFold"
) -> tuple[list, list]:
    """
    데이터셋을 train/validation으로 분리하는 함수.
    
    Args:
        _imagenames (list): 이미지 파일 경로 리스트.
        _labelnames (list): 라벨 파일 경로 리스트.
        groups (list): 그룹 정보 리스트.
        config (dict): 설정 정보. fold 번호 포함.
        is_train (bool): True면 train 데이터를, False면 validation 데이터를 반환.
        split_method (str): 데이터 분리 방법. "GroupKFold", "KFold", "StratifiedKFold" 지원.

    Returns:
        tuple[list, list]: 분리된 이미지와 라벨 리스트.
    """
    dummy_for_groupfold = [0 for _ in _imagenames]  # 그룹 분할을 위한 더미 데이터
    
    if split_method == "GroupKFold":
        splitter = GroupKFold(n_splits=5)
        split_params = (dummy_for_groupfold, groups)
    elif split_method == "KFold":
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        split_params = (dummy_for_groupfold,)

    else:
        raise ValueError(f"Invalid split_method: {split_method}")

    imagenames, labelnames = [], []

    for i, (train_idx, val_idx) in enumerate(splitter.split(_imagenames, *split_params)):
        if mode == 'train':
            if i == config['data']['fold']:
                continue  
            imagenames += list(_imagenames[train_idx])
            labelnames += list(_labelnames[train_idx])
        else:
            if i == config['data']['fold']:
                imagenames += list(_imagenames[val_idx])
                labelnames += list(_labelnames[val_idx])
            break  # validation은 한 번만 사용

    return imagenames, labelnames