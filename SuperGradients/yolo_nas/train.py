import torch

from super_gradients.training import models
from super_gradients.training import Trainer

from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)

import wandb
# Initialize a Weights & Biases run
wandb.init(project="Object_detection", name="yolo_nas_l_os2_e60")

MODEL_ARCH = 'yolo_nas_l'
BATCH_SIZE = 4
MAX_EPOCHS = 60
CHECKPOINT_DIR = f'./checkpoints'
EXPERIMENT_NAME = 'yolo_nas_l_os2_e60'
LOCATION = '../../custom_dataset'
CLASSES = ['General trash', 'Paper', 'Paper pack', 
           'Metal', 'Glass', 'Plastic', 'Styrofoam', 
           'Plastic bag', 'Battery', 'Clothing']

trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)

dataset_params = {
    'data_dir': LOCATION,
    'train_images_dir':'images/train',
    'train_labels_dir':'labels/train',
    'val_images_dir':'images/val',
    'val_labels_dir':'labels/val',
    'test_images_dir':'images/test',
    'classes': CLASSES
}


train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 4
    }
)

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


device = 'cuda' if torch.cuda.is_available() else "cpu"

model = models.get("yolo_nas_l", 
                   num_classes=len(dataset_params['classes']), 
                   pretrained_weights="coco")


train_params = {
    "sg_logger": "wandb_sg_logger", # Weights&Biases Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger for details
    "sg_logger_params":             # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger
      {
        "project_name": "Object_detection", # W&B project name
        "save_checkpoints_remote": True,
        "save_tensorboard_remote": True,
        "save_logs_remote": True,
        "entity": "yolo_nas_m_os2_e70",         # username or team name where you're sending runs
      },
    # ENABLING SILENT MODE
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "SGD",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ckpt_best_name": "ckpt_best",
    "ckpt_name": "ckpt_latest.pt",
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": MAX_EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}
t_len = len(train_data.dataset.transforms)


# 적용된 모든 augementation을 제거
# for i in range(t_len - 1):
#     train_data.dataset.transforms.pop(0)

# DetectionMixup, DetectionHSV, DetectionHorizontalFlip 제거
# train_data.dataset.transforms
# train_data.dataset.transforms.pop(2)
# train_data.dataset.transforms.pop(2)
# train_data.dataset.transforms.pop(2)

# DetectionMosaic,DetectionPaddedRescale input dim 1024x1024로 수정, DetectionRandomAffine target size 1024x1024로 수정
# train_data.dataset.transforms[0].input_dim = (1024,1024)
# train_data.dataset.transforms[0].border_value = 0
# train_data.dataset.transforms[1].target_size = (1024,1024)
# train_data.dataset.transforms[1].degrees = 5
# train_data.dataset.transforms[1].border_value = 0
# train_data.dataset.transforms[1].scale = (1.1, 1.4)
# train_data.dataset.transforms[1].shear = 0.2
# train_data.dataset.transforms[2].input_dim = (1024,1024)
# train_data.dataset.transforms[2].pad_value = 0

# train_data.dataset.plot(plot_transformed_data=True)

trainer.train(	
    model=model, 
    training_params=train_params, 
    train_loader=train_data, 
    valid_loader=val_data
)
