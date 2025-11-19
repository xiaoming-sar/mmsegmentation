crop_size = (1024, 1024)

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size=crop_size),
    std=[58.395, 57.12, 57.375],
    type='SegDataPreProcessor'
)

data_root = '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024_3in1/OASIs_dataset_patch1024_3in1_1024by1024/'
dataset_type = 'SeaObjectDataset'

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, save_optimizer=False, type='CheckpointHook'),
    logger=dict(interval=500, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook')
)

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)

# -------------------- Model: Classic SAM-OCR--------------------
model = dict(
        type='CascadeEncoderDecoder',
        data_preprocessor=data_preprocessor,
        num_stages=2,   
        backbone=dict(
                features_only=True,
                model_name='sam2_hiera_tiny.fb_r896_2pt1',
                # patch_size=7,
                out_indices=(0, 1, 2, 3), 
                pretrained=True,
                type='TIMMBackbone'), 
        neck=dict(
                type='FPN',
                in_channels=[96, 192, 384, 768], # The messy input sizes
                out_channels=256,                # The clean output size
                num_outs=4),                     # 4 scales of features
            
        decode_head=[
                dict(
                    type='FCNHead',
                    in_channels=[256, 256, 256, 256],
                    channels=512,
                    in_index=(0, 1, 2, 3),
                    num_convs=1,
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    concat_input=False,
                    input_transform='resize_concat',
                    dropout_ratio=0.1,
                    num_classes=4,
                    align_corners=False,
                    loss_decode=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,alpha=0.5,loss_weight=0.4, reduction='mean')),
                dict(
                    type='OCRHead',
                    in_channels=[256, 256, 256, 256],  
                    channels=512,
                    ocr_channels=256,
                    input_transform='resize_concat',
                    in_index=(0, 1, 2, 3),
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    dropout_ratio=0.1,
                    num_classes=4,
                    align_corners=False,
                    loss_decode=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,alpha=0.5,loss_weight=1.0, reduction='mean')),
            ],
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
        )



# -------------------- Optimizer --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,               # Works well with large images
        weight_decay=5e-4
    ),
    clip_grad=dict(max_norm=1, norm_type=2)
)

# -------------------- Scheduler --------------------
# param_scheduler = [
#     dict(type='LinearLR', begin=0, end=1000, start_factor=0.001, by_epoch=False),
#     dict(type='MultiStepLR', begin=1000, end=20000, milestones=[16000, 18000], by_epoch=False)
# ]
#60k iters for 
param_scheduler = [
    dict(type='LinearLR', begin=0, end=1000, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=1000, end=40000, milestones=[36000, 38000], by_epoch=False)
]

# -------------------- Training --------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -------------------- Dataloaders --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=10,   # Adjust to fit GPU memory, U-Net lighter than SAM
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        pipeline=train_pipeline,
        ann_file='splits/train.txt'
    )
)

val_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        pipeline=test_pipeline,
        ann_file='splits/val.txt'
    )
)

test_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        pipeline=test_pipeline,
        ann_file='splits/test.txt'
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore', 'mFwIoU'])
test_evaluator = val_evaluator

# -------------------- Misc --------------------
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = '/cluster/projects/nn10004k/packages_install/SAM2.1_Tiny_OCR_40k'
randomness = dict(seed=0)
