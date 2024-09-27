from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class EasyPortraitPSDataset(BaseSegDataset):
    """EasyPortrait dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    METAINFO = dict(
        classes = ('background', 'person'),
    
        palette = [[0, 0, 0], [160, 221, 255]])
    
    def __init__(self, **kwargs):
        super(EasyPortraitPSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        #assert self.file_client.exists(self.img_dir)