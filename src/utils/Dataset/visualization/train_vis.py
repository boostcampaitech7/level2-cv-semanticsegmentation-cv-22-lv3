# visualization.py

import matplotlib.pyplot as plt
import io
import PIL
import wandb
import numpy as np

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label, palette=PALETTE):
    """
    클래스 인덱스를 RGB 이미지로 변환합니다.
    
    Args:
        label (np.ndarray): 클래스 인덱스가 담긴 배열. 형태는 (H, W) 또는 (batch_size, H, W)
        palette (list): 클래스별 RGB 색상 리스트.
        
    Returns:
        np.ndarray: RGB 이미지. 형태는 (H, W, 3) 또는 (batch_size, H, W, 3)
    """
    if label.ndim == 3:
        # 배치가 있는 경우
        batch_size, h, w = label.shape
        rgb = np.zeros((batch_size, h, w, 3), dtype=np.uint8)
        for i in range(batch_size):
            for j, color in enumerate(palette):
                rgb[i][label[i] == j] = color
    elif label.ndim == 2:
        # 배치가 없는 경우
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for j, color in enumerate(palette):
            rgb[label == j] = color
    else:
        raise ValueError("Label array must be 2D or 3D (with batch dimension).")
    return rgb

def visualize_predictions(images, preds, masks, max_visualize=5):
    """
    예측 결과와 실제 마스크를 시각화하여 wandb에 업로드할 이미지를 생성합니다.
    
    Args:
        images (list of torch.Tensor): 시각화할 원본 이미지 리스트. 각 이미지의 형태는 (C, H, W)
        preds (list of torch.Tensor): 예측 마스크 리스트. 각 마스크의 형태는 (H, W)
        masks (list of torch.Tensor): 실제 마스크 리스트. 각 마스크의 형태는 (H, W)
        max_visualize (int): 시각화할 이미지의 최대 개수.
        
    Returns:
        list of wandb.Image: wandb에 업로드할 이미지 리스트.
    """
    figures = []
    num_visualize = min(max_visualize, len(images))
    
    for i in range(num_visualize):
        img = images[i].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        
        # 이미지가 [0,1] 범위라면 [0,255]로 스케일링
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        pred_mask = preds[i].numpy()  # (H, W)
        gt_mask = masks[i].numpy()    # (H, W)
        
        # 팔레트를 이용해 RGB 이미지로 변환
        pred_rgb = label2rgb(pred_mask)
        gt_rgb = label2rgb(gt_mask)

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        # 원본 이미지를 시각화하려면 주석을 해제하세요.
        # axes[0].imshow(img)
        # axes[0].set_title('Image')
        # axes[0].axis('off')

        axes[0].imshow(pred_rgb)
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')

        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # matplotlib Figure를 wandb Image로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb_image = wandb.Image(PIL.Image.open(buf))
        figures.append(wandb_image)
        plt.close(fig)
    
    return figures