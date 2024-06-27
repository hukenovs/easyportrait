from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EasyPortraitDataset(BaseSegDataset):
    """EasyPortrait dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes = ('background', 'person', 'skin',
               'left brow', 'right brow', 'left eye',
               'right eye', 'lips', 'teeth'),
    
        palette=[[0, 0, 0], [223, 87, 188], [160, 221, 255],
               [130, 106, 237], [200, 121, 255], [255, 183, 255],
               [0, 144, 193], [113, 137, 255], [230, 232, 230]])
    
    def __init__(self, **kwargs):
        super(EasyPortraitDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        #assert self.file_client.exists(self.img_dir)