![easyportrait](images/main.jpg)
# EasyPortrait - Face Parsing and Portrait Segmentation Dataset
We introduce a large-scale image dataset **EasyPortrait** for portrait segmentation and face parsing. Proposed dataset can be used in several tasks, such as background removal in conference applications, teeth whitening, face skin enhancement, red eye removal or eye colorization, and so on. 

EasyPortrait dataset size is about **26GB**, and it contains **20 000** RGB images (~17.5K FullHD images) with high quality annotated masks. This dataset is divided into training set, validation set and test set by subject `user_id`. The training set includes 14000 images, the validation set includes 2000 images, and the test set includes 4000 images.

## Downloads

| Link                                     | Size  |
|------------------------------------------|-------|
| [`images`](https://sc.link/YEE9p)         | 26G   |
| [`annotations`](https://sc.link/2kkwv)    | 235M  |
| [`train set`](https://sc.link/Z6696)      | 18.1G |
| [`validation set`](https://sc.link/VOOj1) | 2.6G  |
| [`test set`](https://sc.link/1wwvj)       | 5.2G  |

### Structure
```
.
├── images.zip
│   ├── train/         # Train set: 14k
│   ├── val/           # Validation set: 2k
│   ├── test/          # Test set: 4k
├── annotations.zip
│   ├── meta.zip       # Meta-information (width, height, brightness, imhash, user_id)
│   ├── train/     
│   ├── val/       
│   ├── test/      
...
```

## Models
We provide some pre-trained models as the baseline for portrait segmentation and face parsing. We use mean Intersection over Union (mIoU) as the main metric. 

| Model Name                                     | Parameters (M) | Input shape | mIOU      |
|------------------------------------------------|----------------|-------------|-----------|
| [LR-ASPP + MobileNet-V3](https://sc.link/gBo6) | 1.14           | 1024 × 1024 | 77.55     |
| [FCN + MobileNet-V2](https://sc.link/ErPv)     | 9.71           | 384 × 384   | 74.3      |
| [FCN + MobileNet-V2](https://sc.link/vKjm)     | 9.71           | 512 × 512   | 77.01     |
| [FCN + MobileNet-V2](https://sc.link/9xZ3)     | 9.71           | 1024 × 1024 | 81.23     |
| [FPN + ResNet-50](https://sc.link/6r19)        | 28.5           | 512 × 512   | 83.13     |
| [FPN + ResNet-50](https://sc.link/Gy97)        | 28.5           | 1024 × 1024 | **85.97** |
| [BiSeNet-V2](https://sc.link/ryYE)             | 14.79          | 512 × 512   | 77.93     |
| [BiSeNet-V2](https://sc.link/8wZo)             | 14.79          | 1024 × 1024 | 83.53    |
| [SegFormer-B0](https://sc.link/wMkR)           | 3.72           | 384 × 384   | 79.82     |
| [SegFormer-B0](https://sc.link/0lZX)           | 3.72           | 1024 × 1024 | 84.27     |
| [SegFormer-B2](https://sc.link/AjmO)           | 24.73          | 384 × 384   | 81.59     |
| [SegFormer-B2](https://sc.link/zVnY)           | 24.73          | 512 × 512   | 83.03     |
| [SegFormer-B2](https://sc.link/7vZA)           | 24.73          | 1024 × 1024 | 85.72     |
| [SegFormer-B5](https://sc.link/yQm7)           | 81.97          | 384 × 384   | 81.66     |
| [SegFormer-B5](https://sc.link/xOl9)           | 81.97          | 1024 × 1024 | 85.80     |
| [SegNeXt + MSCAN-T](https://sc.link/Dp0x)      | 4.23           | 384 × 384   | 75.01     |
| [SegNeXt + MSCAN-T](https://sc.link/BlnX)      | 4.23           | 512 × 512   | 78.59     |

## Annotations

Annotations are presented as 2D-arrays, images in `*.png` format with several classes:

| Index | Class      |
|------:|:-----------|
|     0 | BACKGROUND |
|     1 | PERSON     |
|     2 | SKIN       |
|     3 | LEFT BROW  |
|     4 | RIGHT_BROW |
|     5 | LEFT_EYE   |
|     6 | RIGHT_EYE  |
|     7 | LIPS       |
|     8 | TEETH      |

Also, we provide some additional meta-information for dataset in `annotations/meta.zip` file:

|    | attachment_id | user_id | data_hash | width | height | brightness | train | test  | valid |
|---:|:--------------|:--------|:----------|------:|-------:|-----------:|:------|:------|:------|
|  0 | de81cc1c-...  | 1b...   | e8f...    |  1440 |   1920 |        136 | True  | False | False |
|  1 | 3c0cec5a-...  | 64...   | df5...    |  1440 |   1920 |        148 | False | False | True  |
|  2 | d17ca986-...  | cf...   | a69...    |  1920 |   1080 |        140 | False | True  | False |

where:
- `attachment_id` - image file name without extension
- `user_id` - unique anonymized user ID
- `data_hash` - image hash by using Perceptual hashing
- `width` - image width
- `height` - image height
- `brightness` - image brightness
- `train`, `test`, `valid` are the binary columns for train / test / val subsets respectively 

## Images
![easyportrait](images/data.jpg)


## Training, Evaluation and Testing on EasyPortrait

>The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) with 0.30.0 version.

Models were trained and evaluated on 8 NVIDIA V100 GPUs with CUDA 11.2.

For installation process follow the instructions [here](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/get_started.md#installation) and use the **requirements.txt** file in our repository.

### Training
For single GPU mode:
```console
python ./pipelines/tools/train.py ./pipelines/local_configs/easy_portrait_experiments/<model_dir>/<config_file>.py --gpu-id <GPU_ID>
```

For distributed training mode:
```console
./pipelines/tools/dist_train.sh ./pipelines/local_configs/easy_portrait_experiments/<model_dir>/<config_file>.py <NUM_GPUS>
```

### Evaluation
For single GPU mode:
```console
python ./pipelines/tools/test.py <PATH_TO_MODEL_CONFIG>  <PATH_TO_CHECKPOINT> --gpu-id <GPU_ID> --eval mIoU
```

For distributed evaluation mode:
```console
./pipelines/tools/dist_test.sh <PATH_TO_MODEL_CONFIG>  <PATH_TO_CHECKPOINT> <NUM_GPUS> --eval mIoU
```
### Run demo
```console
python ./pipelines/demo/image_demo.py <PATH_TO_IMG> <PATH_TO_MODEL_CONFIG> <PATH_TO_CHECKPOINT> --palette=easy_portrait --out-file=<PATH_TO_OUT_FILE>
```

## Authors and Credits
- [Alexander Kapitanov](https://www.linkedin.com/in/hukenovs)
- [Karina Kvanchiani](https://www.linkedin.com/in/kvanchiani)
- [Sofia Kirillova](https://www.linkedin.com/in/gofixyourself/)

## Citation
You can cite the paper using the following BibTeX entry:

    @article{EasyPortrait,
        title={EasyPortrait - Face Parsing and Portrait Segmentation Dataset},
        author={Kapitanov, Alexander and Kvanchiani, Karina and Kirillova Sofia},
        journal={arXiv preprint <link>},
        year={2023}
    }

## License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf).