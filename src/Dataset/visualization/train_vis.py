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


def label2rgb_multi(label : list) -> list:
    '''
        summary : 다중 클래스 라벨을 RGB로 변경
        args : 라벨 ndarray (클래스 수, 너비, 높이)
        return : RGB로 변경된 ndarray 리스트
    '''
    image_size = label.shape[1:] + (3, ) 
    image = np.zeros(image_size, dtype=np.float32)

    for i in range(label.shape[0]):
        class_mask = label[i] 
        color = np.array(PALETTE[i]) / 255.0 
        color_mask = np.stack([class_mask]*3, axis=-1) * color 
        image += color_mask  

    image = np.clip(image, 0, 1)
    return image


def visualize_predictions(pred, mask, image=None):
    '''
        summary : 예측값과 정답을 시각화
        args : 예측된 마스크 배열, 정답 마스크 배열, 원본 이미지 배열
        return : 시각화 결과 Figure 객체 (예측결과, 원본이미지, TP, FP, Overlay)
    '''
    pred_rgb = label2rgb_multi(pred)
    mask_rgb = label2rgb_multi(mask)
    
    fp = (pred == 1) & (mask == 0)  
    fn = (pred == 0) & (mask == 1)  

    fp_rgb = label2rgb_multi(fp.astype(np.uint8))
    fn_rgb = label2rgb_multi(fn.astype(np.uint8))
    
    if image is not None:
        image = image / 255.0 if image.max() > 1 else image
        overlay = image.copy()
    else:
        overlay = np.zeros_like(pred_rgb)
    
    fp_mask = fp.any(axis=0)
    fn_mask = fn.any(axis=0)
    
    overlay[fp_mask] = [1, 0, 0]  
    overlay[fn_mask] = [0, 0, 1]  
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 12))

    axes[0].imshow(pred_rgb)
    axes[0].set_title('Prediction')
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(fp_rgb)
    axes[2].set_title('False Positives (Pred - GT)')
    axes[2].axis('off')

    axes[3].imshow(fn_rgb)
    axes[3].set_title('False Negatives (GT - Pred)')
    axes[3].axis('off')

    axes[4].imshow(overlay)
    axes[4].set_title('Overlay of FP and FN')
    axes[4].axis('off')

    plt.tight_layout()
    return fig


def save_image_for_visualization(config, masks, preds_to_visualize, outputs, masks_to_visualize):
    '''
        summary : 원하는 갯수의 이미지 시각화를 위해 예측 결과와 정답 마스크를 준비
        args : config파일, 정답 마스크 데이터, 시각화할 에측 결과 저장리스트, 모델의 에측 출력, 시각화할 실제 마스크 저장 리스트
        return : 업데이트 된 preds_to_visiualize 리스트와 masks_to_visalize'리스트 반환
    '''
    if masks.ndim == 3:
        num_classes = len(config.data.classes)
        masks_one_hot = np.eye(num_classes)[masks.numpy()]  
        masks_one_hot = masks_one_hot.transpose(0, 3, 1, 2)  
    else:
        masks_one_hot = masks.numpy()  

    if len(preds_to_visualize) < 5:
        batch_size = outputs.shape[0]
        num_needed = 5 - len(preds_to_visualize)
        num_to_take = min(num_needed, batch_size)
        for i in range(num_to_take):
            preds_to_visualize.append(outputs[i].numpy())
            masks_to_visualize.append(masks_one_hot[i])
    
    return preds_to_visualize, masks_to_visualize