import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def verify_flipped_data(train_dir, sample_size=5, save_samples=True):
    """
    반전된 이미지와 JSON 파일이 올바르게 매칭되고 있는지 검증합니다.
    
    Args:
        train_dir (str): 'train' 디렉토리의 경로
        sample_size (int): 시각적 검증을 위한 샘플 이미지 수
        save_samples (bool): 시각적 검증 이미지를 저장할지 여부
    """
    dcm_dir = os.path.join(train_dir, 'DCM')
    json_dir = os.path.join(train_dir, 'outputs_json')
    
    if not os.path.exists(dcm_dir):
        print(f"DCM 디렉토리를 찾을 수 없습니다: {dcm_dir}")
        return
    
    if not os.path.exists(json_dir):
        print(f"JSON 디렉토리를 찾을 수 없습니다: {json_dir}")
        return
    
    # 'DCM' 디렉토리 내의 모든 'IDxxx_flip' 폴더 목록 가져오기
    id_flip_folders = [folder for folder in os.listdir(dcm_dir) 
                       if os.path.isdir(os.path.join(dcm_dir, folder)) and folder.endswith("_flip")]
    
    if not id_flip_folders:
        print("반전된 ID 폴더가 하나도 존재하지 않습니다.")
    
    print("1. 파일 존재 여부 및 이름 일치 확인:")
    with tqdm(total=len(id_flip_folders), desc="Verifying ID_flip Folders", unit="folder") as pbar_id:
        for id_flip_folder in id_flip_folders:
            original_id = id_flip_folder[:-5]  # '_flip' 제거하여 원본 ID 추출
            dcm_flip_path = os.path.join(dcm_dir, id_flip_folder)
            json_flip_folder = id_flip_folder
            json_flip_path = os.path.join(json_dir, json_flip_folder)
            
            # 반전된 JSON 폴더 확인
            if not os.path.exists(json_flip_path):
                print(f"[Missing] 반전된 JSON 폴더이 없습니다: {json_flip_path}")
            
            # 원본 ID 폴더의 JSON 파일 목록 가져오기
            original_json_dir = os.path.join(json_dir, original_id)
            if not os.path.exists(original_json_dir):
                print(f"[Missing] 원본 JSON 폴더이 없습니다: {original_json_dir}")
                pbar_id.update(1)
                continue
            
            # 원본 이미지 파일 목록
            original_images = [f for f in os.listdir(os.path.join(dcm_dir, original_id)) 
                               if os.path.isfile(os.path.join(dcm_dir, original_id, f)) and 
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img_file in original_images:
                base_name, ext = os.path.splitext(img_file)
                flipped_img_file = f"{base_name}_flip{ext}"
                flipped_img_path = os.path.join(dcm_flip_path, flipped_img_file)
                
                if not os.path.exists(flipped_img_path):
                    print(f"[Missing] 반전된 이미지 파일이 없습니다: {flipped_img_path}")
                
                # JSON 파일 확인
                original_json_file = f"{base_name}.json"
                flipped_json_file = f"{base_name}_flip.json"
                original_json_path = os.path.join(original_json_dir, original_json_file)
                flipped_json_path = os.path.join(json_flip_path, flipped_json_file)
                
                if not os.path.exists(original_json_path):
                    print(f"[Missing] 원본 JSON 파일이 없습니다: {original_json_path}")
                if not os.path.exists(flipped_json_path):
                    print(f"[Missing] 반전된 JSON 파일이 없습니다: {flipped_json_path}")
            
            pbar_id.update(1)
    
    print("\n2. 좌표 반전 정확성 확인:")
    with tqdm(total=len(id_flip_folders), desc="Checking Coordinates", unit="folder") as pbar_id:
        for id_flip_folder in id_flip_folders:
            original_id = id_flip_folder[:-5]  # '_flip' 제거하여 원본 ID 추출
            dcm_flip_path = os.path.join(dcm_dir, id_flip_folder)
            json_flip_folder = id_flip_folder
            json_flip_path = os.path.join(json_dir, json_flip_folder)
            
            # 원본 ID 폴더의 JSON 파일 목록 가져오기
            original_json_dir = os.path.join(json_dir, original_id)
            if not os.path.exists(original_json_dir):
                pbar_id.update(1)
                continue  # 이미 누락된 폴더는 위에서 출력됨
            
            # 원본 이미지 파일 목록
            original_images = [f for f in os.listdir(os.path.join(dcm_dir, original_id)) 
                               if os.path.isfile(os.path.join(dcm_dir, original_id, f)) and 
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img_file in original_images:
                base_name, ext = os.path.splitext(img_file)
                flipped_img_file = f"{base_name}_flip{ext}"
                flipped_json_file = f"{base_name}_flip.json"
                original_json_file = f"{base_name}.json"
                
                original_json_path = os.path.join(original_json_dir, original_json_file)
                flipped_json_path = os.path.join(json_flip_path, flipped_json_file)
                flipped_img_path = os.path.join(dcm_flip_path, flipped_img_file)
                
                if not os.path.exists(original_json_path) or not os.path.exists(flipped_json_path):
                    continue  # 이미 누락된 파일은 위에서 출력됨
                
                try:
                    # 원본 이미지 크기
                    with Image.open(os.path.join(dcm_dir, original_id, img_file)) as img:
                        width, height = img.size
                
                    # 반전된 이미지 크기
                    with Image.open(flipped_img_path) as flipped_img:
                        flipped_width, flipped_height = flipped_img.size
                
                    if width != flipped_width or height != flipped_height:
                        print(f"[Mismatch] 이미지 크기 다름: {img_file}")
                
                    # 원본 JSON 좌표 읽기
                    with open(original_json_path, 'r', encoding='utf-8') as jf:
                        original_data = json.load(jf)
                
                    # 반전된 JSON 좌표 읽기
                    with open(flipped_json_path, 'r', encoding='utf-8') as jf:
                        flipped_data = json.load(jf)
                
                    # 각 annotation의 좌표 비교
                    for orig_ann, flip_ann in zip(original_data.get('annotations', []), flipped_data.get('annotations', [])):
                        orig_points = orig_ann.get('points', [])
                        flip_points = flip_ann.get('points', [])
                        
                        if len(orig_points) != len(flip_points):
                            print(f"[Mismatch] Annotation points count 다름: {img_file}")
                            continue
                        
                        for orig_pt, flip_pt in zip(orig_points, flip_points):
                            orig_x, orig_y = orig_pt
                            flip_x, flip_y = flip_pt
                            expected_flip_x = width - orig_x
                            
                            if flip_x != expected_flip_x or flip_y != orig_y:
                                print(f"[Mismatch] 좌표 반전 오류: {img_file}, 원본 점: {orig_pt}, 반전된 점: {flip_pt}")
                
                except Exception as e:
                    print(f"[Error] 좌표 검증 중 오류 발생: {img_file}")
                    print(e)
            
            pbar_id.update(1)
    
    print("\n3. 시각적 검증:")
    sample_count = 0
    with tqdm(total=sample_size, desc="Visual Verification", unit="sample") as pbar_sample:
        for id_flip_folder in id_flip_folders:
            if sample_count >= sample_size:
                break
            original_id = id_flip_folder[:-5]  # '_flip' 제거하여 원본 ID 추출
            dcm_flip_path = os.path.join(dcm_dir, id_flip_folder)
            json_flip_folder = id_flip_folder
            json_flip_path = os.path.join(json_dir, json_flip_folder)
            
            # 원본 ID 폴더의 JSON 파일 목록 가져오기
            original_json_dir = os.path.join(json_dir, original_id)
            if not os.path.exists(original_json_dir):
                continue  # 이미 누락된 폴더는 위에서 출력됨
            
            # 원본 이미지 파일 목록
            original_images = [f for f in os.listdir(os.path.join(dcm_dir, original_id)) 
                               if os.path.isfile(os.path.join(dcm_dir, original_id, f)) and 
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img_file in original_images:
                if sample_count >= sample_size:
                    break
                base_name, ext = os.path.splitext(img_file)
                flipped_img_file = f"{base_name}_flip{ext}"
                flipped_json_file = f"{base_name}_flip.json"
                original_json_file = f"{base_name}.json"
                
                original_json_path = os.path.join(original_json_dir, original_json_file)
                flipped_json_path = os.path.join(json_flip_path, flipped_json_file)
                flipped_img_path = os.path.join(dcm_flip_path, flipped_img_file)
                
                if not os.path.exists(flipped_json_path) or not os.path.exists(flipped_img_path):
                    continue  # 이미 누락된 파일은 위에서 출력됨
                
                try:
                    # 반전된 이미지 열기
                    with Image.open(flipped_img_path) as flipped_img:
                        flipped_img = flipped_img.convert("RGB")
                        width, height = flipped_img.size
                
                    # 반전된 JSON 열기
                    with open(flipped_json_path, 'r', encoding='utf-8') as jf:
                        flipped_data = json.load(jf)
                
                    # 이미지와 주석 그리기
                    fig, ax = plt.subplots(1)
                    ax.imshow(flipped_img)
                
                    for annotation in flipped_data.get('annotations', []):
                        if annotation.get('type') == 'poly_seg' and 'points' in annotation:
                            points = annotation['points']
                            polygon = patches.Polygon(points, closed=True, edgecolor='red', fill=False, linewidth=2)
                            ax.add_patch(polygon)
                
                    ax.set_title(f"Sample {sample_count+1}: {flipped_img_file}")
                    plt.axis('off')
                
                    if save_samples:
                        # 이미지 저장
                        output_dir = os.path.join(train_dir, 'visual_verification_samples')
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"sample_{sample_count+1}.png")
                        fig.savefig(output_path)
                        plt.close(fig)
                        print(f"저장됨: {output_path}")
                    else:
                        # 이미지 표시
                        plt.show()
                
                    sample_count += 1
                    pbar_sample.update(1)
                
                except Exception as e:
                    print(f"[Error] 시각적 검증 중 오류 발생: {flipped_img_file}")
                    print(e)
    
    print("\n검증이 완료되었습니다.")

if __name__ == "__main__":
    # 원본 데이터 디렉토리 경로 설정
    train_directory = '/data/ephemeral/home/segmentation_baseline/level2-cv-semanticsegmentation-cv-22-lv3/data/train'
    
    # 시각적 검증 이미지를 저장하려면 save_samples=True로 설정
    verify_flipped_data(train_directory, sample_size=5, save_samples=True)
