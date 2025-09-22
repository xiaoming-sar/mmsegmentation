from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

"""
        for MaSTr1325  dataset only replace 4 as 3
            Obstacles and environment = 0 (value zero)
            Water = 1 (value one)
            Sky = 2 (value two)
            Ignore region / unknown category = 4 (value four)
        data_root = '/cluster/projects/nn10004k/ml_SeaObject_Data/MaSTr1325'
        img_dir = 'image'
        ann_dir = 'mask_original'
        ann_dir_new = 'mask'

        if not osp.exists(osp.join(data_root, ann_dir_new)):
            os.makedirs(osp.join(data_root, ann_dir_new))


        import os
        from PIL import Image
        mask_files = os.listdir(osp.join(data_root, ann_dir))
        mask_files = [mask_file for mask_file in mask_files if mask_file.endswith('.png') ]

        for mask_file in mask_files:
            mask = Image.open(osp.join(data_root, ann_dir, mask_file))
            mask = np.array(mask)
            #for MaSTr1325  dataset only replace 4 as 3
            mask[mask == 4] = 3
            mask = Image.fromarray(mask,mode="L")
            mask.save(osp.join(data_root, ann_dir_new, mask_file[:-5] + '_.png'))


        #read the corresponding image of mask_file in img_dir and display the image and mask side by side
        mask_file = mask_files[0]
        mask = Image.open(osp.join(data_root, ann_dir, mask_file))
        mask = np.array(mask)
        img_file = mask_file[:-5] + '.jpg'
        img = Image.open(osp.join(data_root, img_dir, img_file))
        img = np.array(img)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(mask, cmap='gray', vmin=0, vmax=3)
        ax[1].set_title('Mask')
        ax[1].axis('off')
        plt.show()

        # check the shape of the image and mask one by one, see if they are the same, if not print the file name
        for mask_file in mask_files:
            mask = Image.open(osp.join(data_root, ann_dir, mask_file))
            mask = np.array(mask)
            img_file = mask_file[:-5] + '.jpg'
            img = Image.open(osp.join(data_root, img_dir, img_file))
            img = np.array(img)
            if img.shape[:2] != mask.shape:
                print(f'Image and mask shape mismatch: {img_file}, {mask_file}')
            # else:
            #     print(f'Image and mask shape match: {img_file}, {mask_file}')

"""
@DATASETS.register_module()
class MaSTr1325Dataset(BaseSegDataset):
    MAR_CLASSES  = ('Obstacles', 'Sea', 'Sky', 'Others')
    # PALETTE = [[141,211,199], [55, 126, 184], [255,255,179], [251,128,114]]
    PALETTE = [[0,0,0], [50, 50, 50], [100, 100, 100], [150, 150, 150]]
    METAINFO = dict(classes = MAR_CLASSES, palette = PALETTE)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='_.png',
                            reduce_zero_label=False, **kwargs)
