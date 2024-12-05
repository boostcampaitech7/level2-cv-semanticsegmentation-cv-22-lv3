import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.model_loader import model_loader
from Dataset.dataloader import get_test_loader
from model.utils.model_output import get_model_output
from utils.inference_utils import load_model, encode_mask_to_rle, encode_mask_to_rle_gpu, save_to_csv, prepare_inference_environment


def do_inference(config : OmegaConf) -> str:
    '''
    단일 모델을 사용하여 시맨틱 세그멘테이션 추론을 수행합니다.

    summary:
        모델을 로드하고 테스트 데이터셋에 대해 추론을 실행합니다.

    args:
        config : config 파일 경로

    return:
        결과 CSV 파일 경로
    '''
    model = model_loader(config)
    checkpoint_path = config.inference.checkpoint_path
    library = config.model.library
    device, CLASS2IND, IND2CLASS = prepare_inference_environment(config)

    try:  
        model = load_model(model, checkpoint_path, device, library)
        print(f'Checkpoint loaded successfully from: {checkpoint_path}')

    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        raise

    try:
        model = model.to(device)
        model.eval()
        rles = []
        filename_and_class = []
        output_size = tuple(map(int, config.inference.output_size))

        with torch.no_grad():
            test_loader = get_test_loader(config)
            
            for images, image_names in tqdm(test_loader, total=len(test_loader), desc='Inference'):
                images = images.to(device)
                outputs = get_model_output(model, images)
                outputs = F.interpolate(outputs, size=output_size, mode=config.data.valid.interpolate.bilinear)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > config.inference.threshold)

                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        if config.inference.mode == 'cpu':
                            rle = encode_mask_to_rle(segm.cpu().numpy())
                        elif config.inference.mode == 'gpu':
                            rle = encode_mask_to_rle_gpu(segm)
                        rles.append(rle)
                        filename_and_class.append(f'{IND2CLASS[c]}_{image_name}')

        output_path = save_to_csv(filename_and_class, rles, config)
        print(f'Total predictions: {len(rles)}')
        return output_path

    except Exception as e:
        print(f'Error during inference: {e}')
        raise


def main():
    '''
    명령줄 인자를 파싱하고 단일 모델 추론을 실행합니다.

    summary:
        추론을 위한 CLI 및 설정 로드 함수
    '''
    parser = argparse.ArgumentParser(description='Test Semantic Segmentation Model')
    parser.add_argument('--mode', type=str, default='gpu', choices=['cpu', 'gpu'], help='Inference mode: "cpu" or "gpu"')
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pretrained model pt file')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.inference.checkpoint_path = args.checkpoint
    config.inference.mode = args.mode

    output_path = do_inference(config)
    print(f'Inference completed. Results saved to: {output_path}')


if __name__ == '__main__':
    main()