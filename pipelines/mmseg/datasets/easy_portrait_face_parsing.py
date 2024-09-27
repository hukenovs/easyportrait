from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EasyPortraitFPDataset(BaseSegDataset):
    """EasyPortraitFPDataset dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    METAINFO = dict(
    
        classes = ('background', 'skin',
                'left brow', 'right brow', 'left eye',
                'right eye', 'lips', 'teeth'),
        
        palette = [[0, 0, 0], [160, 221, 255],
                [130, 106, 237], [200, 121, 255], [255, 183, 255],
                [0, 144, 193], [113, 137, 255], [230, 232, 230]])
    
    def __init__(self, **kwargs):
        super(EasyPortraitFPDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        #assert self.file_client.exists(self.img_dir)
        
@DATASETS.register_module()
class EasyPortraitFPDatasetCross(BaseSegDataset):
    """EasyPortraitFPDatasetCross dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    METAINFO = dict(
        classes = ('background', 'left brow', 'right brow', 'left eye', 'right eye', 'lips'),
        palette = [[0, 0, 0], [160, 221, 255],
               [130, 106, 237], [200, 121, 255], [255, 183, 255],
               [0, 144, 193]])

    def __init__(self, **kwargs):
        super(EasyPortraitFPDatasetCross, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        #assert self.file_client.exists(self.img_dir)