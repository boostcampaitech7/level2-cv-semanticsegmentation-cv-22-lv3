import os
import pytz
from datetime import datetime
from omegaconf import OmegaConf


class ConfigManager:
    def __init__(self, base_config : str, model_config : str, encoder_config : str = None, save_config : str = None, save_ckpt : str = None) -> None:
        '''
        summary :
            여러 구성 파일을 로드하고 병합하기 위한 ConfigManager 클래스를 초기화합니다.

        args :
            base_config : 기본 config 파일의 경로
            model_config : 모델 config 파일의 경로
            encoder_config : 인코더 config 파일의 경로. 기본값은 None.
            save_config : 구성 파일 config 경로. 기본값은 None.
            save_ckpt : 체크포인트 저장 경로. 기본값은 None.
        
        return :
            반환값이 없습니다.
        '''
        self.base_config = OmegaConf.load(base_config)
        self.model_config = OmegaConf.load(model_config)
        self.encoder_config = OmegaConf.load(encoder_config) if encoder_config else {}
        save_defaults = self.base_config.save if hasattr(self.base_config, 'save') else {}
        self.save = {
            'save_config': save_config if save_config else save_defaults.get('save_config'),
            'save_ckpt': save_ckpt if save_ckpt else save_defaults.get('save_ckpt'),
        }

        return


    def load_config(self) -> OmegaConf:
        '''
        summary :
            구성 파일들을 병합하고, 설정 및 체크포인트를 저장한 후 병합된 구성을 반환합니다.

        args :
            인자값이 없습니다.

        return :
            병합된 최종 config 객체
        '''
        self.base_config.save = self.save

        merged_config = self._merge_config(self.base_config,
                          self.model_config,
                          self.encoder_config)
        
        self._save_ckpt(merged_config)
        self._save_config(merged_config)

        return merged_config


    def _merge_config(self, *configs : OmegaConf) -> OmegaConf:
        '''
        summary :
            여러 구성 객체를 병합합니다.

        args :
            *configs: 병합할 config 파일들의 경로

        return :
            병합된 config 객체
        '''
        return OmegaConf.merge(*configs)


    def _save_config(self, config : OmegaConf) -> None:
        '''
        summary :
            구성 파일을 타임스탬프, 모델, 인코더 정보를 포함한 디렉터리에 저장합니다.

        args :
            config : config 파일 경로

        return :
            반환값이 없습니다.
        '''
        kst = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        base_model = config.model.base_model
        encoder = config.model.architecture.encoder_name
        epoch = config.data.train.max_epoch
        file_name = f'{base_model}_epoch{epoch}.yaml'
        folder_name = f'{timestamp}_{base_model}_{encoder}_cfg'

        save_dir = os.path.join(config.save.save_config, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        OmegaConf.save(config, save_path)
        print(f'설정 파일이 {save_path}에 저장되었습니다.')

        return


    def _save_ckpt(self, config : OmegaConf) -> None:
        '''
        summary :
            체크포인트 저장 디렉터리를 생성하고 구성 객체에 저장 경로를 설정합니다.

        args :
            config : config 파일 경로

        return :
            반환값이 없습니다.
        '''
        kst = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        base_model = config.model.base_model
        encoder = config.model.architecture.encoder_name
        folder_name = f'{timestamp}_{base_model}_{encoder}_ckpt'
        save_ckpt_dir = os.path.join(config.save.save_ckpt, folder_name)
        os.makedirs(save_ckpt_dir, exist_ok=True)
        config.save['save_ckpt'] = save_ckpt_dir

        return