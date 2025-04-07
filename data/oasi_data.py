

import mmseg
import mmcv
import mmengine
import argparse
from mmengine.runner import Runner
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

#create argparser
# parser = argparse.ArgumentParser(description='Train a model')
# parser.add_argument('--config_file', help='train config file path',default=
#                     '/cluster/home/snf52395/mmsegmentation/data/bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024.py')
# parser.add_argument('--data_root', help='root of the dataset', default=
#                     '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2')
# parser.add_argument('--work_dir', help='the dir to save logs and models', default=
#                     '/cluster/projects/nn10004k/packages_install/seaobject')
# parser.add_argument('--batch_size', help='training batch size per gpu', default=3)
# parser.add_argument('--train_max_iters', help='the dir to save logs and models', default=
#                     '/cluster/projects/nn10004k/packages_install/test_data')

# args = parser.parse_args()

def main():
    data_root = '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/TYPE2'
    mmseg_code_root = '/cluster/home/snf52395/mmsegmentation/configs'
    work_dir = '/cluster/projects/nn10004k/packages_install/sam2_test'
    # checkpoint_dir = '/cluster/projects/nn10004k/packages_install/test_data'


    """
    The defination here is ture and tested on the dataset
    array([  0,  50, 100, 150], dtype=uint8), 4 classes,
    0 - sky 40%
    50 - sea water 52%
    100 - land 4%
    150 - sea obj 4%
    """
    img_dir = 'image'
    ann_dir = 'mask'


    # classes = ('Sky', 'Sea', 'Land', 'SeaObj')
    # palette = [[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150]]
    # import os
    # from PIL import Image
    # mask_files = os.listdir(osp.join(data_root, ann_dir))
    # mask_files = [mask_file for mask_file in mask_files if mask_file.endswith('.png')]
    
    # for mask_file in mask_files:
    #     mask = Image.open(osp.join(data_root, ann_dir, mask_file))
    #     mask = np.array(mask)
    #     #replace [  0,  50, 100, 150] with [0, 1, 2, 3]
    #     mask[mask == 0] = 0
    #     mask[mask == 50] = 1
    #     mask[mask == 100] = 2
    #     mask[mask == 150] = 3
    #     mask = Image.fromarray(mask,mode="L")
    #     mask.save(osp.join(data_root, ann_dir, mask_file[:-4] + '_.png'))

    
    # import matplotlib.patches as mpatches
    # img1 = Image.open(osp.join(data_root, ann_dir, 'type2_000_1.png'))
    # plt.figure(figsize=(8, 6))
    # im = plt.imshow(np.array(img.convert('RGB')))

    # # create a patch (proxy artist) for every color 
    # patches = [mpatches.Patch(color=np.array(palette[i])/255., 
    #                           label=classes[i]) for i in range(len(classes))]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
    #            fontsize='large')

    # split train/val set randomly
    # split_dir = 'splits'
    # train_portion = 0.8
    # mmengine.mkdir_or_exist(osp.join(data_root, split_dir))
    # filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
    #     osp.join(data_root, img_dir), suffix='.png')]
    # with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    #   # select first 4/5 as train set
    #   train_length = int(len(filename_list)*train_portion)
    #   f.writelines(line + '\n' for line in filename_list[:train_length])
    # with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    #   # select last 1/5 as train set
    #   f.writelines(line + '\n' for line in filename_list[train_length:])

    # # define dataset
    # @DATASETS.register_module()
    # class SeaObjectDataset(BaseSegDataset):
    #     METAINFO = dict(classes = classes)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png', seg_map_suffix='_.png',
    #                          reduce_zero_label=False, **kwargs)

    # test_dataset = DATASETS.build(dict(
    #     type='SeaObjectDataset',
    #     data_root=data_root,
    #     data_prefix=dict(img_path='image', seg_map_path='mask'),
    #     # Add other required parameters
    # ))
    # cfg = Config.fromfile(osp.join(mmseg_code_root, 'bisenetv1/bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024.py'))
    cfg = Config.fromfile('/cluster/home/snf52395/mmsegmentation/data/hiera-sam2-base+_SegSegformer_SeaObject-1024x1024.py')
    # print(f'Config:\n{cfg.pretty_text}')

    # cfg.norm_cfg = dict(type='BN', requires_grad=True)
    # cfg.crop_size = (1024, 1024)
    # cfg.model.data_preprocessor.size = cfg.crop_size
    # cfg.model.backbone.norm_cfg = cfg.norm_cfg
    # cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    # cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    # cfg.model.decode_head.num_classes = 4
    # cfg.model.auxiliary_head.num_classes = 4
    # cfg.model.decode_head.loss_decode.class_weight = [2.5, 1.923, 25.0, 25.0] 

    # Modify dataset type and path
    cfg.dataset_type = 'SeaObjectDataset'
    cfg.data_root = data_root

    cfg.train_dataloader.batch_size = 8 # 56 for a100 sam2.1 small
    cfg.val_dataloader.batch_size = 8
    cfg.test_dataloader.batch_size = 8

    cfg.num_workers=4
    
    # cfg.train_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations'),
    #     dict(keep_ratio=True, ratio_range=(0.5, 2.0,),
    #         scale=(2048, 1024), type='RandomResize'),
    #     dict(cat_max_ratio=0.75, crop_size=(1024,1024), type='RandomCrop'),
    #     dict(prob=0.5, type='RandomFlip'),
    #     dict(type='PhotoMetricDistortion'),
    #     dict(type='PackSegInputs')
    #     ]

    # cfg.test_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(keep_ratio=True, scale=( 1024, 1024), type='Resize'),
    #     dict(type='LoadAnnotations'),
    #     dict(type='PackSegInputs')
    # ]

 
    # cfg.train_dataloader.dataset.type = cfg.dataset_type
    # cfg.train_dataloader.dataset.data_root = cfg.data_root
    # cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    # cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    # cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

    # cfg.val_dataloader.dataset.type = cfg.dataset_type
    # cfg.val_dataloader.dataset.data_root = cfg.data_root
    # cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
    # cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    # cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

    # cfg.test_dataloader = cfg.val_dataloader

    # #test the dataset 
    # cfg.train_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations'),
    #     dict(keep_ratio=True, ratio_range=(0.5, 2.0,),
    #         scale=(1024, 1024), type='RandomResize'),
    #     dict(cat_max_ratio=0.75, crop_size=(1024,1024), type='RandomCrop'),
    #     dict(prob=0.5, type='RandomFlip'),
    #     dict(type='PhotoMetricDistortion'),
    #     dict(type='PackSegInputs')
    #     ]
    # dataset_test = SeaObjectDataset(data_root=data_root, data_prefix=cfg.train_dataloader.dataset.data_prefix, 
    #                            test_mode=False, pipeline=cfg.train_pipeline)
    # # Load the pretrained weights
    # cfg.load_from = osp.join(checkpoint_dir, 'bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth')

    # Set up working dir to save files and logs.
    cfg.work_dir = work_dir
    cfg.train_cfg.max_iters = 20000 # max iterations X
    cfg.train_cfg.val_interval = 200 # per X iterations validate the model
    cfg.default_hooks.logger.interval = 50
    cfg.default_hooks.checkpoint.interval = 10000

    # Set seed to facilitate reproducing the result
    cfg['randomness'] = dict(seed=0)



    runner = Runner.from_cfg(cfg)
    # start training
    runner.train()

if __name__ == '__main__':
    main()