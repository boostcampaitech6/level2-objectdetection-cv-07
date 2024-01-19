_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py'
]

# backbone 변경시 final.py 에서 덮어 쓰는 방법
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    # type='MaskRCNN', ## 모델은 cascade_rcnn을 사용하기 때문에 주석처리
    backbone=dict(
        _delete_=True,  ## 기존에 백본을 Resnet을 썼는데 Swin으로 쓰겠다. lr과 같은 다른 config에도 같이 사용이 가능한 인자
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768])
        # [256, 512, 1024, 2048] = 기존의 in_channels . backcone마다 channel이 다르다.    
)