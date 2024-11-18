import os
from omegaconf import OmegaConf
import datetime
import pytz


class ConfigManager:
    def __init__(self, base_config, model_config, encoder_config=None, save_config=None, save_ckpt=None):
        self.base_config = OmegaConf.load(base_config)
        self.model_config = OmegaConf.load(model_config)
        self.encoder_config = OmegaConf.load(encoder_config) if encoder_config else {}
        
        # base_config에서 save 블록 가져오기
        save_defaults = self.base_config.save if hasattr(self.base_config, "save") else {}

        # save_config과 save_ckpt가 주어지지 않으면 base_config의 값을 사용
        self.save = {
            "save_config": save_config if save_config else save_defaults.get("save_config"),
            "save_ckpt": save_ckpt if save_ckpt else save_defaults.get("save_ckpt"),
        }

    def load_config(self):
        """
        병합된 config 반환.
        병합 후 save_config로 저장.
        """
        # base_config에 save 블록 업데이트
        self.base_config.save = self.save

        # base_config와 병합
        merged_config = self._merge_config(self.base_config,
                          self.model_config,
                          self.encoder_config)
        
        self._save_ckpt(merged_config)
        # 병합된 설정을 저장
        self._save_config(merged_config)
        return merged_config

    def _merge_config(self, *configs):
        """
        여러 config를 병합하여 반환합니다.
        """
        return OmegaConf.merge(*configs)

    def _save_config(self, config):
        """
        입력받은 config를 지정된 경로에 저장하는 함수.
        """
        # 모델 이름과 epcoh 정보를 가져오기
        kst = pytz.timezone('Asia/Seoul')
        base_model = config.model.architecture.base_model
        epoch = config.train.max_epoch
        # 파일 이름 생성
        file_name = f"{base_model}_epoch{epoch}.yaml"
        # 폴더 이름 생성
        folder_name = f"{datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')}_{base_model}"
        # 폴더 생성
        save_dir = os.path.join(config.save.save_config, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # config 파일 저장 경로
        save_path = os.path.join(save_dir, file_name)
        OmegaConf.save(config, save_path)
        print(f"설정 파일이 {save_path}에 저장되었습니다.")

    def _save_ckpt(self, config):
        """
        모델 체크포인트를 저장하고 config에 해당 경로 정보를 덮어쓰는 함수.
        """
        # 모델 이름과 epoch 정보가져오기
        kst = pytz.timezone('Asia/Seoul')
        base_model = config.model.architecture.base_model
        folder_name = f"{base_model}_{datetime.datetime.now(kst).strftime('%Y-%m-%d %H-%M-%S')}_ckpt"

        save_ckpt_dir = os.path.join(config.save.save_ckpt, folder_name)
        os.makedirs(save_ckpt_dir, exist_ok=True)

        config.save['save_ckpt'] = save_ckpt_dir
