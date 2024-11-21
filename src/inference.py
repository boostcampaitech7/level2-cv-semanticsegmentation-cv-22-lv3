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


def save_to_csv(filename_and_class, rles, cfg):
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    kst = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    pt_name = cfg.inference.checkpoint_path.split('/')[-1].split('.')[0]
    output_filepath = os.path.join(cfg.inference.output_dir, f'{pt_name}_{timestamp}.csv')


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


def do_inference(cfg, mode):
    set_seed(cfg.seed)
    library = cfg.model.library


    CLASS2IND = {v: i for i, v in enumerate(cfg.data.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}


    model = model_loader(cfg)
    checkpoint_path = cfg.inference.checkpoint_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # try:  
    #     checkpoint = torch.load(checkpoint_path, map_location=device) if library == 'torchvision' else smp.from_pretrained(checkpoint_path)


    #     if isinstance(checkpoint, dict):
    #         if 'state_dict' in checkpoint:
    #             model.load_state_dict(checkpoint['state_dict'])
    #             print("Loaded state_dict from checkpoint.")
    #         else:
    #             model.load_state_dict(checkpoint)
    #             print("Loaded state_dict directly from checkpoint.")
    #     else:
    #         raise ValueError("Unsupported checkpoint format. Expected a dict with 'state_dict' or a state_dict directly.")




    #     print(f"Checkpoint loaded successfully from: {checkpoint_path}")
    # except Exception as e:
    #     print(f"Error loading checkpoint: {e}")
    #     raise


    try:
        model = model.to(device)
        model.eval()
        rles = []
        filename_and_class = []
        output_size = tuple(map(int, cfg.inference.output_size))

        with torch.no_grad():
            test_loader = get_test_loader(cfg)
            for images, image_names in tqdm(test_loader, total=len(test_loader), desc="Inference"):
                images = images.to(device)
                outputs = model(images)


                if isinstance(outputs, dict):
                    outputs = outputs['out']


                outputs = F.interpolate(outputs, size=output_size, mode='bilinear')
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > cfg.inference.threshold)


                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        if mode == 'cpu':
                            rle = encode_mask_to_rle(segm.cpu().numpy())
                        elif mode == 'gpu':
                            rle = encode_mask_to_rle_gpu(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")


        output_path = save_to_csv(filename_and_class, rles, cfg)
        print(f"Total predictions: {len(rles)}")
        return output_path


    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Semantic Segmentation Model")
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help="Inference mode: 'cpu' or 'gpu'")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the experiment config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fcn_resnet50_best_model.pt', help='Path to the pretrained model pt file')


    args = parser.parse_args()


    cfg = OmegaConf.load(args.config)
    cfg.inference.checkpoint_path = args.checkpoint


    output_path = do_inference(cfg, args.mode)
    print(f"Inference completed. Results saved to: {output_path}")
