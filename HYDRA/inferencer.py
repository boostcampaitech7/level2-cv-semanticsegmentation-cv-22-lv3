# inferencer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
from utils import encode_mask_to_rle, encode_mask_to_rle_gpu
import ttach as tta  # ttach 라이브러리 임포트

class Inferencer:
    def __init__(self, model, model_name, device, config, threshold=0.5, output_size=(2048, 2048)):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.threshold = threshold
        self.output_size = output_size

        # 모델이 Segformer인 경우, 커스텀 TTA Wrapper를 사용합니다.
        if model_name == 'segformer':
            from torch import nn
            class SegformerTTAModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, images):
                    outputs = self.model(pixel_values=images)
                    logits = outputs.logits
                    return logits
            self.model = SegformerTTAModelWrapper(model)
        else:
            self.model = model

        # TTA 적용 여부 확인
        self.use_tta = config.inference.get('use_tta', False)

        if self.use_tta:
            # TTA 변환 정의 (config.yaml에서 지정한 변환 사용)
            tta_transform_list = []
            for transform_name in config.inference.transforms:
                if hasattr(tta, transform_name):
                    transform_class = getattr(tta, transform_name)
                    # 특정 변환에 필요한 추가 파라미터가 있을 경우, 여기에 추가 처리 필요
                    if transform_name == 'Rotate90':
                        # Rotate90은 angles 파라미터가 필요합니다.
                        # config.yaml에서 angles를 지정할 수 있도록 확장할 수 있습니다.
                        # 현재 예시에서는 기본 각도를 사용합니다.
                        tta_transform_list.append(tta.Rotate90(angles=[0, 90, 180, 270]))
                    else:
                        tta_transform_list.append(transform_class())
                else:
                    raise ValueError(f"TTA transform {transform_name} is not supported by ttach.")

            # TTA 변환 조합
            tta_transforms = tta.Compose(tta_transform_list)

            # TTA를 적용한 모델 래핑
            self.tta_model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode='mean')

            # 모델을 평가 모드로 전환
            self.tta_model.eval()
            self.tta_model.to(self.device)
        else:
            # TTA를 사용하지 않는 경우, 원본 모델을 그대로 사용
            self.model.eval()
            self.model.to(self.device)

    def inference(self, test_loader):
        try:
            rles = []
            filename_and_class = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Inference" + (" with TTA" if self.use_tta else "")):
                    images, image_names = batch
                    images = images.to(self.device)

                    if self.use_tta:
                        # TTA 모델을 사용하여 추론
                        outputs = self.tta_model(images)
                    else:
                        # 원본 모델을 사용하여 추론
                        if self.model_name == 'segformer':
                            outputs = self.model(images)
                        else:
                            model_output = self.model(images)
                            if isinstance(model_output, dict) and 'out' in model_output:
                                outputs = model_output['out']
                            else:
                                outputs = model_output

                    # 출력 크기 조정
                    outputs = F.interpolate(outputs, size=self.output_size, mode='bilinear', align_corners=False)
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > self.threshold)
                    
                    for output, image_name in zip(outputs, image_names):
                        for c, segm in enumerate(output):
                            class_name = self.config.IND2CLASS[c]
                            if segm.sum() == 0:
                                rle = ''  # RLE 값이 없을 경우 빈 문자열로 설정
                            else:
                                if self.config.inference_to_gpu is True:
                                    rle = encode_mask_to_rle_gpu(segm)
                                else:
                                    segm = segm.cpu().numpy()
                                    rle = encode_mask_to_rle(segm)
                            rles.append(rle)
                            filename_and_class.append(f"{class_name}_{image_name}")
            return rles, filename_and_class
        except Exception as e:
            print(f"추론 중 오류가 발생했습니다: {e}")
            raise

    def save_results(self, rles, filename_and_class, output_path):
        try:
            # 클래스와 파일 이름을 분리
            classes, filenames = zip(*[x.split('_', 1) for x in filename_and_class])  # '_'으로 한 번만 split
            # os.path.basename을 사용하여 디렉토리 경로 제거
            image_names = [os.path.basename(f) for f in filenames]
            # 데이터프레임 생성
            df = pd.DataFrame({'image_name': image_names, 'class': classes, 'rle': rles})
            df.to_csv(output_path, index=False)
            print(f"Inference results saved to {output_path}")
        except Exception as e:
            print(f"결과를 저장하는 중 오류가 발생했습니다: {e}")
            raise
