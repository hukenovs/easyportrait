![easyportrait](images/main.jpg)
# EasyPortrait - Face Parsing and Portrait Segmentation Dataset
We introduce a large-scale image dataset **EasyPortrait** for portrait segmentation and face parsing. Proposed dataset can be used in several tasks, such as background removal in conference applications, teeth whitening, face skin enhancement, red eye removal or eye colorization, and so on. 

EasyPortrait dataset size is about **91.78GB**, and it contains **40,000** RGB images (~38.3K FullHD images) with high quality annotated masks. This dataset is divided into training set, validation set and test set by subject `user_id`. The training set includes 14000 images, the validation set includes 4,000 images, and the test set includes 6,000 images.

For more information see our paper [EasyPortrait – Face Parsing and Portrait Segmentation Dataset](https://arxiv.org/abs/2304.13509).

## 🔥 Changelog
 - **`2023/11/13`**: We release EasyPortrait 2.0. ✌️
   - **40,000** RGB images (~38.3K FullHD images) 
   - Added diversity by region, race, human emotions and lighting conditions
   - The data was further cleared and new ones were added
   - Train/val/test split: (30,000) **75%** / (4,000) **10%** / (6,000) **15%** by subject `user_id`
   - Multi-gpu training and testing
   - Added new models for face parsing and portrait segmentation
   - Dataset size is **91.78GB**
   - **13,705** unique persons
 - **`2023/02/23`**: EasyPortrait (Initial Dataset) 💪
   - Dataset size is **26GB**
   - **20,000** RGB images (~17.5K FullHD images) with **9** classes annotated
   - Train/val/test split: (14,000) **70%** / (2,000) **10%** / (4,000) **20%** by subject `user_id`
   - **8,377** unique persons
   <!-- - The distance is 0.5 to 4 meters from the camera -->

 Old EasyPortrait dataset is also available into branch `EasyPortrait_v1`! 
## Downloads

| Link                                                                                                          | Size  |
|---------------------------------------------------------------------------------------------------------------|-------|
| [`images`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/images.zip)           | 91.8 GB  |
| [`annotations`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/annotations.zip) | 657.1 MB  |
| [`meta`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/meta.zip) | 1.9 MB  |
| [`train set`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/train.zip)         | 68.3 GB |
| [`validation set`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/val.zip)      | 10.7 GB  |
| [`test set`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/v2/test.zip)           | 12.8 GB  |

Also, you can download EasyPortrait dataset from [Kaggle](https://www.kaggle.com/datasets/kapitanov/easyportrait).

### Structure
```
.
├── images.zip
│   ├── train/         # Train set: 30k
│   ├── val/           # Validation set: 4k
│   ├── test/          # Test set: 6k
├── annotations.zip
│   ├── train/     
│   ├── val/       
│   ├── test/      
├── meta.zip       # Meta-information (width, height, brightness, imhash, user_id)
...
```

## Models
We provide some pre-trained models as the baseline for portrait segmentation and face parsing. We use mean Intersection over Union (mIoU) as the main metric. 

#### Portrait segmentation:
| Model Name                                     | Parameters (M) | Input shape | mIoU      |
|------------------------------------------------|----------------|-------------|-----------|
| [BiSeNet-V2](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/bisenet-ps.pth)                       | 56.5          | 384 x 384 | 97.95     |
| [DANet](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/danet-ps.pth)                    | 190.2           | 384 x 384   | 98.63    |
| [DeepLabv3](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/deeplabv3-ps.pth)                    | 260           | 384 x 384   | 98.63    |
| [ExtremeC3Net](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/extremenet-ps.pth)                    | 0.15           | 384 x 384   | 96.54    |
| [Fast SCNN](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fcn_scnn-ps.pth)               | 6.13          | 384 x 384 | 97.64 |
| [FCN + MobileNetv2](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fcn-ps.pth)               | 31.17           | 384 x 384 | 98.19 |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-ps-1024.pth)                 | 108.91           | 1024 × 1024   | 98.54    |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-ps-512.pth)               | 108.91            | 512 × 512 | 98.64 |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-ps.pth)               | 108.91          | 384 x 384 | 98.64 |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-ps-224.pth)               | 108.91          | 224 × 224 | 98.31 |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps-1024.pth)  | 14.9          | 1024 × 1024 |98.74     |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps-512.pth)            | 14.9           | 512 × 512   | 98.66      |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps.pth)          | 14.9           | 384 x 384 | 98.61     |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps-224.pth)            | 14.9           | 224 × 224   | 98.17     |
| [SINet](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/sinet-ps.pth)                    | 0.13          | 384 x 384   | 93.32  |


#### Face parsing:
| Model Name                                     | Parameters (M) | Input shape | mIoU      |
|------------------------------------------------|----------------|-------------|-----------|
| [BiSeNet-V2](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/bisenet-fp.pth)                       | 56.5          | 384 x 384 | 76.72     |
| [DANet](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/danet-fp.pth)                    | 190.2           | 384 x 384   | 79.3    |
| [DeepLabv3](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/deeplabv3-fp.pth)                    | 260           | 384 x 384   | 79.11    |
| [EHANet](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/ehanet-ps.pth)                    | 44.81          | 384 x 384   | 72.56    |
| [Fast SCNN](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fcn_scnn-ps.pth)               | 6.13          | 384 x 384 | 67.56|
| [FCN + MobileNetv2](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fcn-fp.pth)               | 31.17           | 384 x 384 | 75.23 |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-fp-1024.pth)                 | 108.91           | 1024 × 1024   | 85.37   |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-fp-512.pth)               | 108.91            | 512 × 512 | 83.33 |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-fp.pth)               | 108.91          | 384 x 384 | 81.83  |
| [FPN + ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/fpn-fp-224.pth)               | 108.91          | 224 × 224 | 75.6 |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-fp-1024.pth)  | 14.9          | 1024 × 1024 |85.42     |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-fp-512.pth)            | 14.9           | 512 × 512   | 83.19      |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-fp.pth)          | 14.9           | 384 x 384 | 81.38    |
| [SegFormer-B0](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-fp-224.pth)            | 14.9           | 224 × 224   | 74.83     |


## Annotations

Annotations are presented as 2D-arrays, images in `*.png` format with several classes:

| Index | Class      |
|------:|:-----------|
|     0 | BACKGROUND |
|     1 | PERSON     |
|     2 | SKIN       |
|     3 | LEFT_BROW  |
|     4 | RIGHT_BROW |
|     5 | LEFT_EYE   |
|     6 | RIGHT_EYE  |
|     7 | LIPS       |
|     8 | TEETH      |

Also, we provide some additional meta-information for dataset in `annotations/meta.zip` file:

|    | image_name | user_id | height | width | set | brightness |
|---:|:--------------|:--------|:----------|------:|-------:|-----------:|
|  0 | a753e021-...  | 56...   | 720    |  960 |   train |        126 | 
|  1 | 4ff04492-...  | ba...   | 1920    |  1440 |   test |        173 | 
|  2 | e8934c99-...  | 1d...   | 1920    |  1440 |   val |        187 |

where:
- `image_name` - image file name without extension
- `user_id` - unique anonymized user ID
- `height` - image height
- `width` - image width
- `brightness` - image brightness
- `set` - "train", "test" or "val" for train / test / val subsets respectively 

## Images
![easyportrait](images/data.jpg)


## Training, Evaluation and Testing on EasyPortrait

>The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) with 0.30.0 version.

Models were trained and evaluated on 8 NVIDIA V100 GPUs with CUDA 11.2.

For installation process follow the instructions [here](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/get_started.md#installation) and use the **requirements.txt** file in our repository.

<details>
  <summary>Training</summary>

  For single GPU mode:
  ```console
  python ./pipelines/tools/train.py ./pipelines/local_configs/easy_portrait_experiments/<model_dir>/<config_file>.py --gpu-id <GPU_ID>
  ```

  For distributed training mode:
  ```console
  ./pipelines/tools/dist_train.sh ./pipelines/local_configs/easy_portrait_experiments/<model_dir>/<config_file>.py <NUM_GPUS>
  ```
</details>

<details>
  <summary>Evaluation</summary>

  For single GPU mode:
  ```console
  python ./pipelines/tools/test.py <PATH_TO_MODEL_CONFIG>  <PATH_TO_CHECKPOINT> --gpu-id <GPU_ID> --eval mIoU
  ```

  For distributed evaluation mode:
  ```console
  ./pipelines/tools/dist_test.sh <PATH_TO_MODEL_CONFIG>  <PATH_TO_CHECKPOINT> <NUM_GPUS> --eval mIoU
  ```
</details>

<details>
  <summary>Run demo</summary>

  ```console
  python ./pipelines/demo/image_demo.py <PATH_TO_IMG> <PATH_TO_MODEL_CONFIG> <PATH_TO_CHECKPOINT> --palette=easy_portrait --out-file=<PATH_TO_OUT_FILE>
  ```
</details>

## Authors and Credits
- [Alexander Kapitanov](https://www.linkedin.com/in/hukenovs)
- [Karina Kvanchiani](https://www.linkedin.com/in/kvanchiani)
- [Elizaveta Petrova](https://www.linkedin.com/in/kleinsbotle)
- [Karen Efremyan](https://www.linkedin.com/in/befozg)
- [Alexander Sautin](https://www.linkedin.com/in/befozg/alexander-sautin-b5039623b)

## Links
- [arXiv](https://arxiv.org/abs/2304.13509)
- [Paperswithcode](https://paperswithcode.com/dataset/easyportrait)
- [Kaggle](https://www.kaggle.com/datasets/kapitanov/easyportrait)
- [Habr](https://habr.com/ru/companies/sberdevices/articles/731794/)
- [Gitlab](https://gitlab.aicloud.sbercloud.ru/rndcv/easyportrait)

## Citation
You can cite the paper using the following BibTeX entry:

    @article{EasyPortrait,
        title={EasyPortrait - Face Parsing and Portrait Segmentation Dataset},
        author={Kapitanov, Alexander and Kvanchiani, Karina and Kirillova Sofia},
        journal={arXiv preprint arXiv:2304.13509},
        year={2023}
    }

## License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf).
