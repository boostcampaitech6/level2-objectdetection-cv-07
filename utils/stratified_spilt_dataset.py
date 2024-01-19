import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import argparse


path = os.path.dirname(os.path.abspath(__file__))

# dataset 디렉토리가 level2-objectdetection-cv-07 디렉토리와 같은 레벨에 있을 때
data_dir = os.path.join(path, '..', '..', 'dataset')
anns_path = os.path.join(data_dir, 'os_train.json')


def main(args):
    with open(anns_path) as f: data = json.load(f)

    images = data['images']
    anns = data['annotations']
    categories = data['categories']
    info = data['info']
    licenses = data['licenses']

    var = [(ann['image_id'], ann['category_id']) for ann in anns]
    X = np.ones((len(anns),1))
    y = np.array([v[1] for v in var]) # ann['category_id']
    groups = np.array([v[0] for v in var]) # ann['image_id']

    cv = StratifiedGroupKFold(n_splits=args.n_split, shuffle=True, random_state=411)

    path = args.path

    if not os.path.exists(path):
        os.makedirs(path)

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):

        # 중복된 image_id를 제거하기 위해 set으로 변환
        train_image_idxs = set(groups[train_idx])
        val_image_idxs = set(groups[val_idx])

        # train과 val 데이터셋의 images와 annotations를 재구성
        train_images = [img for img in images if img["id"] in train_image_idxs]
        train_anns = [ann for ann in anns if ann["id"] in train_idx]

        val_images = [img for img in images if img["id"] in val_image_idxs]
        val_anns = [ann for ann in anns if ann["id"] in val_idx]

        # json 파일로 저장하기 위한 dictionary 생성
        train_data = {
                "info": info,
                "licenses": licenses,
                "images": train_images,
                "categories": categories,
                "annotations": train_anns
                }
        
        val_data = {
                "info": info,
                "licenses": licenses,
                "images": val_images,
                "categories": categories,
                "annotations": val_anns
                }
        

        train_dir = os.path.join(path, f'cv_train_skfold{i + 1}.json')
        val_dir = os.path.join(path, f'cv_val_skfold{i + 1}.json')

        # 생성한 dictionary를 json 파일로 저장
        with open(train_dir, 'w') as f:
            json.dump(train_data, f)

        with open(val_dir, 'w') as f:   
            json.dump(val_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset 디렉토리가 level2-objectdetection-cv-07 디렉토리와 같은 레벨에 있을 때
    parser.add_argument('--path', 
                        '-p', 
                        type=str, 
                        default=os.path.join(path, '..', '..', 'dataset', 'skfold'))
    
    parser.add_argument('--n_split', 
                        '-n', 
                        type=int, default='5')
    
    args = parser.parse_args()
    main(args)
