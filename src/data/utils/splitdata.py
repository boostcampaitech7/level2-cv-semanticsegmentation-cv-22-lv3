from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, KFold


def split_data(_imagenames: list, _labelnames: list, groups: list, config: DictConfig, 
               mode: str = 'train', split_method: str = 'GroupKFold') -> tuple[list, list]:
    '''
        summary : 
            split_method에 맞는 방식으로 Train/Val을 n_split 갯수만큼 분할합니다.
        args
            _imagenames : 이미지 이름 리스트
            _labelnames : 라벨 이름 리스트들
            grouops : GroupKFold를 위한 더미 데이터
            config : method, nssplit 정보가 담긴 파일
            mode : 학습 모드 설정 
            split_method : 분할 방법 
            
        returns
            train, val에 맞게 설정된 (image, label) 튜플 리스트
    '''
    if split_method == 'GroupKFold':
        n_splits = config.data.get('n_splits', config.data.splits)
        fold = config.data.get('fold', 0)
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(_imagenames, groups=groups))
        
    elif split_method == 'KFold':
        n_splits = config.data.get('n_splits', config.data.splits)
        shuffle = config.data.get('shuffle', True)
        random_state = config.data.get('random_state', 42)
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(splitter.split(_imagenames))
    else:
        raise ValueError(f'Invalid split_method: {split_method}')
    if config.data.fold >= len(splits):
        raise ValueError(f'Fold number {config.data.fold} is out of range for {len(splits)} splits.')
    train_idx, val_idx = splits[config.data.fold]

    if mode == 'train':
        imagenames = [ _imagenames[idx] for idx in train_idx ]
        labelnames = [ _labelnames[idx] for idx in train_idx ]
    elif mode == 'val':
        imagenames = [ _imagenames[idx] for idx in val_idx ]
        labelnames = [ _labelnames[idx] for idx in val_idx ]
    else:
        raise ValueError('Invalide mod chose train or val')
    
    return imagenames, labelnames