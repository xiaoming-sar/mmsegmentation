norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (
    1024,
    1024,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size=(
        1024,
        1024,
    )),
    type='SegDataPreProcessor')
data_root = '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2'
dataset_type = 'SeaObjectDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=50, type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    backbone=dict(
        type='HieraSAM2',
        # img_size=1024,
        init_cfg=dict(
            checkpoint=
            '/cluster/projects/nn10004k/packages_install/sam_checkpoints/sam2.1_hiera_base_plus_ConvMMseg.pt',
            prefix='backbone',
            type='Pretrained'),
        # out_channels=-1, # not channel reduction and all 768 channel will be reserved for decoder
        # out_indices=[2, 5, 8, 11],
        # patch_size=16,
        # use_abs_pos=True,
        # use_rel_pos=True,
        # window_sizeSSS=14
        ),

     decode_head=[
        dict(
            type='FCNHead',
            in_channels=768, # from 1024 to 256
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.5,
                reduction='mean',class_weight=None)), # class_weight=[2.5, 1.923, 25.0, 25.0]
        dict(
            type='OCRHead',
            in_channels=768, # from 1024 to 768
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                 type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.5,
                reduction='mean', class_weight=None))
              ],

    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0005),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=1000,
        end=80000,
        milestones=[60000, 72000],
        by_epoch=False,
    )
]

randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        data_root=
        '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SeaObjectDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=300, type='IterBasedTrainLoop', val_interval=50)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        data_root=
        '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    1024,
                    1024,
                ),
                type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='SeaObjectDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        1024,
        1024,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='image', seg_map_path='mask'),
        data_root=
        '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SeaObjectDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/cluster/projects/nn10004k/packages_install/seaobject'
