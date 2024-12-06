import pandas as pd
import argparse


def merge_and_replace_rle(csv1_path: str, csv2_path: str, output_path: str, classes: list) -> None:
    '''
    summary :
        두 개의 CSV 파일을 병합하고, 특정 클래스의 RLE 값을 지정된 CSV 파일에 덮어씌웁니다.

    args : 
        csv1_path : 첫 번째 CSV 파일의 경로
        csv2_path : 두 번째 CSV 파일의 경로
        output_path : 병합된 결과를 저장할 CSV 파일의 경로
        classes : RLE 값을 대체할 클래스 목록

    return :
        반환값이 없습니다. 결과 CSV 파일을 지정된 경로에 저장합니다.
    '''

    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)

    print(f'len(csv1): {len(csv1)}, len(csv2): {len(csv2)}')
    print(
        f'Number of rows with different order: '
        f'{sum(csv1['image_name'] != csv2['image_name'])}'
    )
    
    for i in classes: 
        print(
            f'Number of images with different predictions for class "{i}": '
            f'{sum(csv1[csv1['class'] == i]['rle'] != csv2[csv2['class'] == i]['rle'])}'
        )
    
    for num in range(2):
        file_1, file_2 = csv1, csv2
        if num == 1:
            file_1, file_2 = csv2, csv1
        if file_1['rle'].isna().any(): 
            image_name = file_1[file_1['rle'].isna()]['image_name']
            cls_name = file_1[file_1['rle'].isna()]['class']
            for i, j in zip(image_name, cls_name): 
                rle = list(file_2[(file_2['image_name']==i) & (file_2['class']==j)]['rle'])
                file_1.loc[(file_1['image_name'] == i) & (file_1['class'] == j), 'rle'] = rle


    csv2_radius = csv2[csv2['class'].isin(classes)]

    merged = csv1.merge(
        csv2_radius[['image_name', 'class', 'rle']],
        on=['image_name', 'class'],
        how='left',
        suffixes=('', '_csv2')
    )

    merged['rle'] = merged['rle_csv2'].combine_first(merged['rle'])
    merged.drop(columns=['rle_csv2'], inplace=True)
    merged.to_csv(output_path, index=False)
    print(f'Merged CSV file has been saved to "{output_path}".')

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge csv files')
    parser.add_argument('--csv1', type=str, default='../../results/SJH_SR.csv', 
                        help='Base CSV file path')
    parser.add_argument('--csv2', type=str, default='../../results/unet_effib7_aug.csv', 
                        help='CSV file path to take RLE values for specific classes')
    parser.add_argument('--output', type=str, default='../../results/merged_result2.csv',
                        help='Path to save the merged CSV file')
    parser.add_argument('--classes', type=list, default=['Radius'], 
                        help='List of classes to take RLE values from csv2')
    args = parser.parse_args()

    merge_and_replace_rle(args.csv1, args.csv2, args.output, args.classes)
