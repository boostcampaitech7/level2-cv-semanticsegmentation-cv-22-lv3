import pandas as pd
import argparse


def merge_and_replace_rle(csv1_path: str, csv2_path: str, output_path: str, classes: list) -> None:
    """
    두 개의 CSV 파일을 병합하고, csv2에서 특정 클래스의 RLE 값을 csv1에
    덮어씌웁니다. 결과로 생성된 CSV 파일을 저장합니다.

    Args:
        csv1_path (str): 첫 번째 CSV 파일의 경로.
        csv2_path (str): 두 번째 CSV 파일의 경로.
        output_path (str): 병합된 결과를 저장할 CSV 파일의 경로.

    Returns:
        None
    """

    # CSV 파일 읽기
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)

    # 데이터 길이 및 진단 정보 출력
    print(f"len(csv1): {len(csv1)}, len(csv2): {len(csv2)}")
    print(
        f"순서가 다른 행의 개수: "
        f"{sum(csv1['image_name'] != csv2['image_name'])}"
    )
    
    for i in classes: 
        print(
            f"'{i}' 클래스의 예측값이 다른 이미지 개수: "
            f"{sum(csv1[csv1['class'] == i]['rle'] != csv2[csv2['class'] == i]['rle'])}"
        )
    
    # NA값 확인
    for num in range(2):
        file_1, file_2 = csv1, csv2
        if num == 1:
            file_1, file_2 = csv2, csv1
        if file_1['rle'].isna().any(): 
            image_name = file_1[file_1['rle'].isna()]['image_name']
            cls_name = file_1[file_1['rle'].isna()]['class']
            for i, j in zip(image_name, cls_name): 
                rle = file_2[(file_2['image_name']==i) & (file_2['class']==j)]['rle']
                file_1[(file_1['image_name']==i) & (file_1['class']==j)] = rle


    # csv2에서 특정 클래스(classes)만 필터링
    csv2_radius = csv2[csv2['class'].isin(classes)]

    # csv1과 필터링된 csv2 데이터를 병합
    merged = csv1.merge(
        csv2_radius[['image_name', 'class', 'rle']],
        on=['image_name', 'class'],
        how='left',
        suffixes=('', '_csv2')
    )

    # csv2에서 가져온 rle 값을 적용
    merged['rle'] = merged['rle_csv2'].combine_first(merged['rle'])

    # 병합 과정에서 생성된 임시 열 제거
    merged.drop(columns=['rle_csv2'], inplace=True)

    # 병합된 DataFrame을 CSV 파일로 저장
    merged.to_csv(output_path, index=False)
    print(f"병합된 CSV 파일이 '{output_path}'에 저장되었습니다.")

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge csv files")
    parser.add_argument('--csv1', type=str, default="../../results/SJH_SR.csv", 
                        help="기반이 될 csv 경로")
    parser.add_argument('--csv2', type=str, default="../../results/unet_effib7_aug.csv", 
                        help='특정 class의 rle값을 가져올 csv 경로')
    parser.add_argument('--output', type=str, default="../../results/merged_result2.csv",
                        help='병합된 csv를 저장할 경로')
    parser.add_argument('--classes', type=list, default=['Radius'], 
                        help='csv2에서 가져올 rle값의 클래스를 담은 리스트')
    args = parser.parse_args()

    # 함수 실행
    merge_and_replace_rle(args.csv1, args.csv2, args.output, args.classes)
