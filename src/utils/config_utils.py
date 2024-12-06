import os
import pytz
from datetime import datetime
from omegaconf import OmegaConf


class ConfigManager:
    def __init__(self, base_config : str, model_config : str, encoder_config : str = None) -> None:
        '''
        summary :
            여러 구성 파일을 로드하고 병합하기 위한 ConfigManager 클래스를 초기화합니다.

        args :
            base_config : 기본 config 파일의 경로
            model_config : 모델 config 파일의 경로
            encoder_config : 인코더 config 파일의 경로. 기본값은 None.
            checkpoint : 체크포인트 파일의 경로. 기본값은 None
        '''
        self.base_config = OmegaConf.load(base_config)
        self.model_config = OmegaConf.load(model_config)
        self.encoder_config = OmegaConf.load(encoder_config) if encoder_config else {}
        self.save_dir = self.base_config.save.save_dir


    def load_config(self) -> OmegaConf:
        '''
        summary :
            구성 파일들을 병합하고, 설정 및 체크포인트를 저장한 후 병합된 구성을 반환합니다.

        args :
            인자값이 없습니다.

        return :
            병합된 최종 config 객체
        '''
        merged_config = self._merge_config(self.base_config,
                          self.model_config,
                          self.encoder_config)
        
        run_dir = self._create_run_directory(merged_config)
        merged_config.save.save_dir = run_dir
        self._save_ckpt(merged_config, run_dir)
        self._save_output(merged_config, run_dir)
        self._save_config(merged_config, run_dir)

        return merged_config

    @staticmethod
    def update_config(config : OmegaConf, key: str, value) -> None:
        '''
        구성 파일의 특정 키를 업데이트하고 저장합니다.

        Args:
            config : config 파일의 경로
            key (str) : 업데이트할 구성 키
            value : 업데이트할 값
        
        retrun:
            반환값이 없습니다.
        '''
        config_path = config.save.save_dir

        try:
            OmegaConf.update(config, key, value)
            print(f'구성 키 {key}가 {value}로 업데이트되었습니다.')
        except Exception as e:
            print(f'구성 업데이트 중 오류가 발생했습니다.: {e}')
        
        ConfigManager._save_config(config, config_path)

        return None


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


    def _create_run_directory(self, config: OmegaConf) -> str:
        '''
        summary :
            타임스탬프와 모델 이름을 기반으로 실행 디렉토리를 생성합니다.

        args :
            config : 병합된 config 객체

        return :
            생성된 실행 디렉토리 경로
        '''
        kst = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        base_model = config.model.base_model
        
        if self.encoder_config:
            encoder_name = self.encoder_config.model.architecture.encoder_name
            folder_name = f'{timestamp}_{base_model}_{encoder_name}'
        else:
            folder_name = f'{timestamp}_{base_model}'
            
        save_dir = os.path.join(config.save.save_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir
    
    
    @staticmethod
    def _save_config(config : OmegaConf, run_dir: str) -> None:
        '''
        summary :
            구성 파일을 타임스탬프, 모델, 인코더 정보를 포함한 디렉터리에 저장합니다.

        args :
            config : config 파일 경로
            run_dir : config 파일이 저장될 경로

        return :
            반환값이 없습니다.
        '''
        base_model = config.model.base_model
        file_name = f'{base_model}_config.yaml'
        cfg_dir = os.path.join(run_dir, file_name)
        OmegaConf.save(config, cfg_dir)
        print(f'설정 파일이 {cfg_dir}에 저장되었습니다.')

        return None


    def _save_ckpt(self, config : OmegaConf, run_dir : str) -> None:
        '''
        summary :
            체크포인트 저장 디렉터리를 생성하고 구성 객체에 저장 경로를 설정합니다.

        args :
            config : config 파일 경로
            run_dir : config 파일이 저장될 경로
            
        return :
            반환값이 없습니다.
        '''
        ckpt_dir = os.path.join(run_dir, 'checkpoint')
        os.makedirs(ckpt_dir, exist_ok=True)
        config.save.save_ckpt = ckpt_dir
        print(f'체크포인트 저장 디렉토리가 {ckpt_dir}에 생성되었습니다')

        return None
    

    def _save_output(self, config : OmegaConf, run_dir : str) -> None:
        '''
        summary:
            csv 저장 디렉터리를 생성하고 구성 객체에 저장 경로를 설정합니다.
        
        args :
            config : config 파일 경로
            run_dir : config 파일이 저장될 경로
            
        return :
            반환값이 없습니다.
        '''
        output_dir = os.path.join(run_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        config.save.output_dir = output_dir
        print(f'output 저장 디렉토리가 {output_dir}에 생성되었습니다.')

        return None