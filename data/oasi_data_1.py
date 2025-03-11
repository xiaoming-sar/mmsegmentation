

# import mmseg
# import mmcv
# import mmengine
from mmengine import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from PIL import Image
import matplotlib.patches as mpatches
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

def main():
    data_root = '/cluster/projects/nn10004k/packages_install/test_data/iccv09Data'
    mmseg_code_root = '/cluster/home/snf52395/mmsegmentation'
    work_dir = '/cluster/projects/nn10004k/packages_install/seaobject'
    checkpoint_dir = '/cluster/projects/nn10004k/packages_install/test_data'


    """
    array([  0,  50, 100, 150], dtype=uint8), 4 classes,
    0 - sky 40%
    50 - sea water 52%
    100 - land 4%
    150 - sea obj 4%
    """
    img_dir = 'images'
    ann_dir = 'labels'


    # img = mmcv.imread(osp.join(data_root, img_dir,'6000124.jpg'))
    # plt.figure(figsize=(8, 6))
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()

    # # define classes
    classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
            [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

    # for file in mmengine.scandir(osp.join(data_root, ann_dir), suffix='.regions.txt'):
    #   seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    #   seg_img = Image.fromarray(seg_map).convert('P')
    #   seg_img.putpalette(np.array(palette, dtype=np.uint8))
    #   seg_img.save(osp.join(data_root, ann_dir, file.replace('.regions.txt', 
    #                                                          '.png')))
    # img = Image.open(osp.join(data_root, ann_dir,'6000124.png')) 
    # plt.figure(figsize=(8, 6))
    # # im = plt.imshow(np.array(img.convert('RGB')))
    # im = plt.imshow(np.array(img))
    # # create a patch (proxy artist) for every color 
    # patches = [mpatches.Patch(color=np.array(palette[i])/255., 
    #                           label=classes[i]) for i in range(8)]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
    #            fontsize='large')
    # plt.show()


    # # split train/val set randomly
    # split_dir = 'splits'
    # mmengine.mkdir_or_exist(osp.join(data_root, split_dir))
    # filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
    #     osp.join(data_root, ann_dir), suffix='.png')]
    # with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    #   # select first 4/5 as train set
    #   train_length = int(len(filename_list)*4/5)
    #   f.writelines(line + '\n' for line in filename_list[:train_length])
    # with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    #   # select last 1/5 as train set
    #   f.writelines(line + '\n' for line in filename_list[train_length:])

    # define dataset
    @DATASETS.register_module()
    class SeaObjectDataset(BaseSegDataset):
        METAINFO = dict(classes = classes, palette = palette)
        def __init__(self, **kwargs):
            super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

    cfg = Config.fromfile(osp.join(mmseg_code_root, 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'))
    # print(f'Config:\n{cfg.pretty_text}')

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.crop_size = (256, 256)
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 8
    cfg.model.decode_head.loss_decode.class_weight = [1.0] * cfg.model.decode_head.num_classes + [0.1]
    cfg.model.auxiliary_head.num_classes = 8

    # Modify dataset type and path
    cfg.dataset_type = 'SeaObjectDataset'
    cfg.data_root = data_root

    cfg.train_dataloader.batch_size = 2

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(320, 240), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]


    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

    cfg.test_dataloader = cfg.val_dataloader


    # Load the pretrained weights
    cfg.load_from = osp.join(checkpoint_dir, 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')

    # Set up working dir to save files and logs.
    cfg.work_dir = work_dir

    cfg.train_cfg.max_iters = 20
    cfg.train_cfg.val_interval = 200
    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = 200

    # Set seed to facilitate reproducing the result
    cfg['randomness'] = dict(seed=0)

    # Let's have a look at the final config used for training
    # print(f'Config:\n{cfg.pretty_text}')


    runner = Runner.from_cfg(cfg)
    # start training
    runner.train()

if __name__ == '__main__':
    main()