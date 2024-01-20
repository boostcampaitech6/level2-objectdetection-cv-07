# 라이브러리 및 모듈 import
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import pandas as pd
from tqdm import tqdm
import argparse

# CustomDataset class 선언

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds()[index]

        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # 라벨 등 이미지 외 다른 정보 없기 때문에 train dataset과 달리 이미지만 전처리

        # transform
        if self.transforms:
            sample = self.transforms(image=image)

        return sample['image'], image_id

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

# Albumentation을 이용, augmentation 선언
def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ])


def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

from effdet import DetBenchPredict
import gc

# Effdet config를 통해 모델 불러오기 + ckpt load
def load_net(checkpoint_path, device):
    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = 10
    config.image_size = (512,512)

    config.soft_nms = False
    config.max_det_per_image = 25

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)

# valid function
def valid_fn(val_data_loader, model, device):
    outputs = []
    for images, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = torch.stack(images) # bs, ch, w, h
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4],
                            'scores': out.detach().cpu().numpy()[:,4],
                            'labels': out.detach().cpu().numpy()[:,-1]})
    return outputs

def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    annotation = '../dataset/test.json'
    data_dir = '../dataset'
    val_dataset = CustomDataset(annotation, data_dir, get_valid_transform())
    epoch = args.epoch
    checkpoint_path = f'epoch_{epoch}_tf_d2_ap.pth'
    score_threshold = 0.5
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(device)

    model = load_net(checkpoint_path, device)

    outputs = valid_fn(val_data_loader, model, device)

    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(abs(box[0]*2)) + ' ' + str(abs(
                    box[1]*2)) + ' ' + str(abs(box[2]*2)) + ' ' + str(abs(box[3]*2)) + ' '
                print(box, score, label)
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'submission_{epoch}_tf_d4_ap.csv', index=None)
    print(submission.head())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int, default='5')
    args = parser.parse_args()

    main(args)
