import numpy as np
import matplotlib.pyplot as plt

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb_multi(label):

    image_size = label.shape[1:] + (3, ) 
    image = np.zeros(image_size, dtype=np.float32)

    for i in range(label.shape[0]):
        class_mask = label[i] 
        color = np.array(PALETTE[i]) / 255.0 
        color_mask = np.stack([class_mask]*3, axis=-1) * color 
        image += color_mask  

    image = np.clip(image, 0, 1)
    return image

def visualize_predictions(pred, mask):

    pred_rgb = label2rgb_multi(pred)
    mask_rgb = label2rgb_multi(mask)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    axes[0].imshow(pred_rgb)
    axes[0].set_title('Prediction')
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    plt.tight_layout()
    return fig