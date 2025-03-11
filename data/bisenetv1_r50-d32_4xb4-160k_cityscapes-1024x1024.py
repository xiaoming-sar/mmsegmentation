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
    type='SegDataPreProcessor')
data_root = '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2'
dataset_type = 'SeaObjectDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, type='CheckpointHook'),
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
    auxiliary_head=[
        dict(
            channels=256,
            concat_input=False,
            in_channels=512,
            in_index=1,
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=4,
            num_convs=1,
            type='FCNHead'),
        dict(
            channels=256,
            concat_input=False,
            in_channels=512,
            in_index=2,
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=4,
            num_convs=1,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        backbone_cfg=dict(
            contract_dilation=True,
            depth=50,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            in_channels=3,
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            norm_eval=False,
            num_stages=4,
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            strides=(
                1,
                2,
                2,
                2,
            ),
            style='pytorch',
            type='ResNet'),
        context_channels=(
            512,
            1024,
            2048,
        ),
        in_channels=3,
        init_cfg=None,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        out_channels=1024,
        out_indices=(
            0,
            1,
            2,
        ),
        spatial_channels=(
            256,
            256,
            256,
            512,
        ),
        type='BiSeNetV1'),
    data_preprocessor=dict(
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
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=0,
        loss_decode=dict(
            class_weight=[
                2.5,
                1.923,
                25.0,
                25.0,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=4,
        num_convs=1,
        type='FCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.05, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.05, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
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
train_cfg = dict(max_iters=300, type='IterBasedTrainLoop', val_interval=200)
train_dataloader = dict(
    batch_size=3,
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
                    2048,
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
            2048,
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
