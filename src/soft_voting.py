import os
import pytz
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from omegaconf import OmegaConf
from train.trainer import set_seed
import segmentation_models_pytorch as smp
from model.model_loader import model_loader
from Dataset.dataloader import get_test_loader
from model.utils.model_output import get_model_output

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def encode_mask_to_rle_gpu(mask):
    mask = mask.to(torch.uint8)
    pixels = mask.flatten()
    zeros = torch.zeros(1, dtype=pixels.dtype, device=pixels.device)
    pixels = torch.cat([zeros, pixels, zeros])
    changes = (pixels[1:] != pixels[:-1]).nonzero().squeeze() + 1
    runs = changes.clone()
    runs[1::2] = runs[1::2] - runs[::2]
    runs = runs.cpu().numpy()
    return ' '.join(str(x) for x in runs)

def save_to_csv(filename_and_class, rles):
    # KST 시간대 설정
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

    # 결과 파일 저장 경로 설정 (project/results 폴더)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project 디렉토리 경로
    output_folder = os.path.join(base_dir, 'results')  # results 폴더 경로
    os.makedirs(output_folder, exist_ok=True)  # results 폴더가 없으면 생성
    output_filepath = os.path.join(output_folder, f'{timestamp}_soft_voting_ensembled.csv')

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(output_filepath, index=False)
    print(f"Submission file saved to: {output_filepath}")
    return output_filepath

def load_model(model, checkpoint_path, device, library):
    checkpoint = torch.load(checkpoint_path, map_location=device) if library == 'torchvision' else smp.from_pretrained(checkpoint_path)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    print(f"Checkpoint loaded successfully from: {checkpoint_path}")
    return model

def do_inference(configs):
    set_seed(configs[0].seed)
    
    # Ensure all configs have the same number of classes and order
    CLASS2IND = {v: i for i, v in enumerate(configs[0].data.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    # Prepare device
    device = torch.device('cuda') if configs[0].inference.mode == 'gpu' else torch.device('cpu')
    
    # Load all models
    models = []
    for cfg in configs:
        model = model_loader(cfg)
        model = load_model(model, cfg.inference.checkpoint_path, device, cfg.model.library)
        model = model.to(device)
        model.eval()
        models.append(model)

    # Prepare output size and test loader (using the first config)
    output_size = tuple(map(int, configs[0].inference.output_size))
    test_loader = get_test_loader(configs[0])

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for images, image_names in tqdm(test_loader, total=len(test_loader), desc="Ensemble Inference"):
            images = images.to(device)
            
            # Collect model outputs
            all_model_outputs = []
            for model in models:
                outputs = get_model_output(model, images)
                outputs = F.interpolate(outputs, size=output_size, mode=configs[0].data.valid.interpolate.bilinear)
                outputs = torch.sigmoid(outputs)
                all_model_outputs.append(outputs)
            
            # Soft voting (average probabilities)
            ensemble_outputs = torch.mean(torch.stack(all_model_outputs), dim=0)
            ensemble_outputs = (ensemble_outputs > cfg.inference.threshold)

            for output, image_name in zip(ensemble_outputs, image_names):
                for c, segm in enumerate(output):
                    if configs[0].inference.mode == 'cpu':
                        rle = encode_mask_to_rle(segm.cpu().numpy())
                    elif configs[0].inference.mode == 'gpu':
                        rle = encode_mask_to_rle_gpu(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    output_path = save_to_csv(filename_and_class, rles)
    print(f"Total predictions: {len(rles)}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Semantic Segmentation Model with Ensemble")
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help="Inference mode: 'cpu' or 'gpu'")
    parser.add_argument('--configs', type=str, nargs='+', required=True, help='Paths to multiple config files')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='Paths to multiple pretrained model pt files')

    args = parser.parse_args()

    # Validate input
    if len(args.configs) != len(args.checkpoints):
        raise ValueError("Number of configs must match number of checkpoints")

    # Load configs
    configs = []
    for config_path, checkpoint_path in zip(args.configs, args.checkpoints):
        cfg = OmegaConf.load(config_path)
        cfg.inference.checkpoint_path = checkpoint_path
        cfg.inference.mode = args.mode
        configs.append(cfg)

    output_path = do_inference(configs)
    print(f"Ensemble inference completed. Results saved to: {output_path}")