import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import OmegaConf
from train.trainer import set_seed
from model.model_loader import model_loader
from Dataset.dataloader import get_test_loader
from model.utils.model_output import get_model_output
from utils.inference_utils import load_model, encode_mask_to_rle, encode_mask_to_rle_gpu, save_to_csv


def do_inference(cfg):
    set_seed(cfg.seed)
    library = cfg.model.library

    CLASS2IND = {v: i for i, v in enumerate(cfg.data.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}


    model = model_loader(cfg)

    checkpoint_path = cfg.inference.checkpoint_path
    device = torch.device('cuda') if cfg.inference.mode == 'gpu' else torch.device('cpu')

    try:  
        model = load_model(model, checkpoint_path, device, library)    # 모델 로드

        print(f"Checkpoint loaded successfully from: {checkpoint_path}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

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
                outputs = get_model_output(model, images)

                outputs = F.interpolate(outputs, size=output_size, mode=cfg.data.valid.interpolate.bilinear)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > cfg.inference.threshold)

                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        if cfg.inference.mode == 'cpu':
                            rle = encode_mask_to_rle(segm.cpu().numpy())
                        elif cfg.inference.mode == 'gpu':
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
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pretrained model pt file')

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.inference.checkpoint_path = args.checkpoint
    cfg.inference.mode = args.mode

    output_path = do_inference(cfg)
    print(f"Inference completed. Results saved to: {output_path}")
