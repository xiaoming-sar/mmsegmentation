from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

"""
SeaObjectDataset is a subclass of BaseSegDataset
can be created from orignial dataset 
    # import os
    # from PIL import Image
    # mask_files = os.listdir(osp.join(data_root, ann_dir))
    # mask_files = [mask_file for mask_file in mask_files if mask_file.endswith('.png')]
    
    # for mask_file in mask_files:
    #     mask = Image.open(osp.join(data_root, ann_dir, mask_file))
    #     mask = np.array(mask)
    #     #replace [  0,  50, 100, 150] with [0, 1, 2, 3]
    #     mask[mask == 0] = 0 # sky and vessel
    #     mask[mask == 50] = 1 # Sea
    #     mask[mask == 100] = 2 # Land
    #     mask[mask == 150] = 3 # SeaObj
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


"""
@DATASETS.register_module()
class SeaObjectDataset(BaseSegDataset):
    MAR_CLASSES  = ('Others', 'Sea', 'Land', 'SeaObj')
    # PALETTE = [[141,211,199], [55, 126, 184], [255,255,179], [251,128,114]]
    PALETTE = [[0,0,0], [50, 50, 50], [100, 100, 100], [150, 150, 150]]
    METAINFO = dict(classes = MAR_CLASSES, palette = PALETTE)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='_.png',
                            reduce_zero_label=False, **kwargs)
