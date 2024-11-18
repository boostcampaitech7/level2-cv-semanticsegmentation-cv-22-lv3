import os
import pytz
import datetime
from omegaconf import OmegaConf


class ConfigManager:
    def __init__(self, base_config, model_config, encoder_config=None, save_config=None, save_ckpt=None):
        self.base_config = OmegaConf.load(base_config)
        self.model_config = OmegaConf.load(model_config)
        self.encoder_config = OmegaConf.load(encoder_config) if encoder_config else {}
        

        save_defaults = self.base_config.save if hasattr(self.base_config, "save") else {}


        self.save = {
            "save_config": save_config if save_config else save_defaults.get("save_config"),
            "save_ckpt": save_ckpt if save_ckpt else save_defaults.get("save_ckpt"),
        }


    def load_config(self):
        self.base_config.save = self.save


        merged_config = self._merge_config(self.base_config,
                          self.model_config,
                          self.encoder_config)
        
        self._save_ckpt(merged_config)
        self._save_config(merged_config)
        return merged_config


    def _merge_config(self, *configs):
        return OmegaConf.merge(*configs)


    def _save_config(self, config):
        kst = pytz.timezone('Asia/Seoul')
        base_model = config.model.architecture.base_model
        epoch = config.train.max_epoch
        file_name = f"{base_model}_epoch{epoch}.yaml"
        folder_name = f"{datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')}_{base_model}"
        save_dir = os.path.join(config.save.save_config, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        

        save_path = os.path.join(save_dir, file_name)
        OmegaConf.save(config, save_path)
        print(f"설정 파일이 {save_path}에 저장되었습니다.")


    def _save_ckpt(self, config):
        kst = pytz.timezone('Asia/Seoul')
        base_model = config.model.architecture.base_model
        folder_name = f"{base_model}_{datetime.datetime.now(kst).strftime('%Y-%m-%d %H-%M-%S')}_ckpt"
        save_ckpt_dir = os.path.join(config.save.save_ckpt, folder_name)
        os.makedirs(save_ckpt_dir, exist_ok=True)
        config.save['save_ckpt'] = save_ckpt_dir
