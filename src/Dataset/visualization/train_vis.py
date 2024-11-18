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
    if label.ndim == 3:
        batch_size, h, w = label.shape
        rgb = np.zeros((batch_size, h, w, 3), dtype=np.uint8)
        for i in range(batch_size):
            for j, color in enumerate(palette):
                rgb[i][label[i] == j] = color


    elif label.ndim == 2:
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for j, color in enumerate(palette):
            rgb[label == j] = color


    else:
        raise ValueError("Label array must be 2D or 3D (with batch dimension).")
    

    return rgb


def visualize_predictions(images, preds, masks, max_visualize=5):
    '''
        예측 결과와 실제 마스크를 시각화하여 wandb에 업로드할 이미지를 생성합니다.
        
        Args:
            images (list of torch.Tensor): 시각화할 원본 이미지 리스트. 각 이미지의 형태는 (C, H, W)
            preds (list of torch.Tensor): 예측 마스크 리스트. 각 마스크의 형태는 (H, W)
            masks (list of torch.Tensor): 실제 마스크 리스트. 각 마스크의 형태는 (H, W)
            max_visualize (int): 시각화할 이미지의 최대 개수.
            
        Returns:
            list of wandb.Image: wandb에 업로드할 이미지 리스트.
    '''
    figures = []
    num_visualize = min(max_visualize, len(images))
    
    for i in range(num_visualize):
        img = images[i].numpy().transpose(1, 2, 0)  
        
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        pred_mask = preds[i].numpy()  
        gt_mask = masks[i].numpy()   
        
        
        pred_rgb = label2rgb(pred_mask)
        gt_rgb = label2rgb(gt_mask)

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        

        axes[0].imshow(pred_rgb)
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')


        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb_image = wandb.Image(PIL.Image.open(buf))
        figures.append(wandb_image)
        plt.close(fig)
    

    return figures