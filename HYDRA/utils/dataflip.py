import os
import json
from PIL import Image

def flip_images_and_json(train_dir):
    """
    주어진 train 디렉토리 내의 DCM 폴더에서 각 ID 폴더의 이미지를 좌우 반전하여
    해당 ID_flip 폴더에 저장하고, 대응되는 JSON 파일의 좌표도 반전하여 저장합니다.
    
    Args:
        train_dir (str): 'train' 디렉토리의 경로
    """
    # 디렉토리 경로 설정
    dcm_dir = os.path.join(train_dir, 'DCM')
    json_dir = os.path.join(train_dir, 'outputs_json')
    
    if not os.path.exists(dcm_dir):
        print(f"DCM 디렉토리를 찾을 수 없습니다: {dcm_dir}")
        return
    
    if not os.path.exists(json_dir):
        print(f"JSON 디렉토리를 찾을 수 없습니다: {json_dir}")
        return
    
    # DCM 디렉토리 내의 모든 ID 폴더를 탐색
    for id_folder in os.listdir(dcm_dir):
        id_folder_path = os.path.join(dcm_dir, id_folder)
        
        if not os.path.isdir(id_folder_path):
            continue  # 폴더가 아니면 건너뜀
        
        # _flip 폴더 이름 생성
        flip_folder = f"{id_folder}_flip"
        flip_folder_path = os.path.join(dcm_dir, flip_folder)
        
        # _flip 폴더가 없으면 생성
        if not os.path.exists(flip_folder_path):
            os.makedirs(flip_folder_path)
            print(f"생성됨: {flip_folder_path}")
        
        # JSON의 _flip 폴더 생성
        json_flip_folder = f"{id_folder}_flip"
        json_flip_folder_path = os.path.join(json_dir, json_flip_folder)
        if not os.path.exists(json_flip_folder_path):
            os.makedirs(json_flip_folder_path)
            print(f"생성됨: {json_flip_folder_path}")
        
        # ID 폴더 내의 모든 이미지 파일을 탐색
        for img_file in os.listdir(id_folder_path):
            img_path = os.path.join(id_folder_path, img_file)
            
            if not os.path.isfile(img_path):
                continue  # 파일이 아니면 건너뜀
            
            # 이미지 파일인지 확인 (확장자 기준)
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue  # 이미지 파일이 아니면 건너뜀
            
            try:
                # 이미지 열기
                with Image.open(img_path) as img:
                    # 좌우 반전
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # 저장할 파일 경로 (예: image_R_flip.png)
                    base_name, ext = os.path.splitext(img_file)
                    flipped_img_name = f"{base_name}_flip{ext}"
                    flipped_img_path = os.path.join(flip_folder_path, flipped_img_name)
                    
                    # 이미지 저장
                    flipped_img.save(flipped_img_path)
                    print(f"저장됨: {flipped_img_path}")
                    
                    # 이미지의 폭(width) 얻기
                    width, height = img.size
                
                # 대응되는 JSON 파일 경로
                json_file = f"{base_name}.json"
                json_path = os.path.join(json_dir, id_folder, json_file)
                
                if not os.path.exists(json_path):
                    print(f"JSON 파일을 찾을 수 없습니다: {json_path}")
                    continue
                
                # JSON 파일 열기
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                
                # 모든 annotations의 좌표를 반전
                for annotation in data.get('annotations', []):
                    if 'points' in annotation:
                        flipped_points = []
                        for point in annotation['points']:
                            x, y = point
                            new_x = width - x
                            flipped_points.append([new_x, y])
                        annotation['points'] = flipped_points
                
                # 반전된 JSON 파일 저장 경로
                flipped_json_path = os.path.join(json_flip_folder_path, f"{base_name}_flip.json")
                
                # 반전된 JSON 저장
                with open(flipped_json_path, 'w', encoding='utf-8') as jf:
                    json.dump(data, jf, ensure_ascii=False, indent=4)
                print(f"저장됨: {flipped_json_path}")
            
            except Exception as e:
                print(f"처리 중 오류 발생: {img_path}")
                print(e)

if __name__ == "__main__":
    # 원본 데이터 디렉토리 경로 설정
    # 예: '/data/ephemeral/home/segmentation_baseline/level2-cv-semanticsegmentation-cv-22-lv3/data/train'
    train_directory = '/data/ephemeral/home/segmentation_baseline/level2-cv-semanticsegmentation-cv-22-lv3/data/train'
    
    flip_images_and_json(train_directory)
