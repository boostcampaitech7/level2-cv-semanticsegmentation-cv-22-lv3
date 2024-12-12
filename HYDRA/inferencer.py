# inferencer.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os
from utils import encode_mask_to_rle, encode_mask_to_rle_gpu
import ttach as tta

class Inferencer:
    """
    학습된 모델을 이용하여 추론을 수행합니다.
    TTA 적용 가능, 결과를 RLE로 인코딩 후 CSV 저장 기능 포함.
    """
    def __init__(self, model, model_name, device, config, threshold=0.5, output_size=(2048, 2048)):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.threshold = threshold
        self.output_size = output_size

        # Wrap Segformer if needed
        if model_name == 'segformer':
            from torch import nn
            class SegformerTTAModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, images):
                    outputs = self.model(pixel_values=images)
                    return outputs.logits
            self.model = SegformerTTAModelWrapper(model)
        else:
            self.model = model

        self.use_tta = config.inference.get('use_tta', False)

        if self.use_tta:
            tta_transforms = []
            for transform_name in config.inference.transforms:
                if hasattr(tta, transform_name):
                    transform_class = getattr(tta, transform_name)
                    if transform_name == 'Rotate90':
                        tta_transforms.append(tta.Rotate90(angles=[0, 90, 180, 270]))
                    else:
                        tta_transforms.append(transform_class())
                else:
                    raise ValueError(f"TTA transform {transform_name} not supported.")

            tta_compose = tta.Compose(tta_transforms)
            self.tta_model = tta.SegmentationTTAWrapper(self.model, tta_compose, merge_mode='mean')
            self.tta_model.eval()
            self.tta_model.to(self.device)
        else:
            self.model.eval()
            self.model.to(self.device)

    def inference(self, test_loader):
        """
        테스트 데이터에 대해 추론을 수행하고 RLE 결과를 반환합니다.
        """
        try:
            rles = []
            filename_and_class = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Inference" + (" with TTA" if self.use_tta else "")):
                    images, image_names = batch
                    images = images.to(self.device)

                    if self.use_tta:
                        outputs = self.tta_model(images)
                    else:
                        if self.model_name == 'segformer':
                            outputs = self.model(images)
                        else:
                            model_output = self.model(images)
                            outputs = model_output['out'] if isinstance(model_output, dict) and 'out' in model_output else model_output

                    outputs = F.interpolate(outputs, size=self.output_size, mode='bilinear', align_corners=False)
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > self.threshold)

                    for output, image_name in zip(outputs, image_names):
                        for c, segm in enumerate(output):
                            class_name = self.config.IND2CLASS[c]
                            if segm.sum() == 0:
                                rle = ''
                            else:
                                rle = encode_mask_to_rle_gpu(segm) if self.config.inference_to_gpu else encode_mask_to_rle(segm.cpu().numpy())
                            rles.append(rle)
                            filename_and_class.append(f"{class_name}_{image_name}")
            return rles, filename_and_class
        except Exception as e:
            print(f"Error occurred during inference: {e}")
            raise

    def save_results(self, rles, filename_and_class, output_path):
        """
        RLE 결과를 CSV로 저장합니다.
        """
        try:
            classes, filenames = zip(*[x.split('_', 1) for x in filename_and_class])
            image_names = [os.path.basename(f) for f in filenames]
            df = pd.DataFrame({'image_name': image_names, 'class': classes, 'rle': rles})
            df.to_csv(output_path, index=False)
            print(f"Inference results saved to {output_path}")
        except Exception as e:
            print(f"Error occurred while saving results: {e}")
            raise
