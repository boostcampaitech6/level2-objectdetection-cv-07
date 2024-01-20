from super_gradients.training import models
from super_gradients.training import Trainer

from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val

import pandas as pd
import glob

MODEL_ARCH = 'yolo_nas_l'
BATCH_SIZE = 32
MAX_EPOCHS = 100
CHECKPOINT_DIR = f'./checkpoints'
EXPERIMENT_NAME = 'yolo_nas_l'
LOCATION = '../../custom_dataset'
CLASSES = ['General trash', 'Paper', 'Paper pack', 
           'Metal', 'Glass', 'Plastic', 'Styrofoam', 
           'Plastic bag', 'Battery', 'Clothing']

dataset_params = {
    'data_dir': LOCATION,
    'train_images_dir':'images/train',
    'train_labels_dir':'labels/train',
    'val_images_dir':'images/val',
    'val_labels_dir':'labels/val',
    'test_images_dir':'images/test',
    'classes': CLASSES
}

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)

trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="/checkpoints/yolo_nas_l_os2_e60/RUN_20240116_190106_218185/ckpt_best.pth"
                        ).cuda()

source = '../../custom_dataset/images/test'
all = glob.glob(source + '/*jpg')
all.sort()

results = best_model.predict(all, fuse_model=False)

all_data = []

# yolo2voc
for id, image_prediction in enumerate(results):
    prediction_strings = []
    img = f"test/{id:04d}.jpg"

    class_names = image_prediction.class_names
    labels = image_prediction.prediction.labels
    confidence = image_prediction.prediction.confidence
    bboxes = image_prediction.prediction.bboxes_xyxy

    for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
        x_center, y_center, width, height = bbox
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        xmin, ymin, xmax, ymax = bbox
        prediction_strings.append(f"{int(label)} {conf:.6f} {xmin} {ymin} {xmax} {ymax}")

    all_data.append({"PredictionString": " ".join(prediction_strings), "image_id": img})

result_df = pd.DataFrame(all_data)
result_df.to_csv("yolo_nas_l.csv", index=False)