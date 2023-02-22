![easyportrait](images/main.jpg)
# EasyPortrait - Face Parsing and Portrait Segmentation Dataset
We introduce a large-scale image dataset **EasyPortrait** for portrait segmentation and face parsing. Proposed dataset can be used in several tasks, such as background removal in conference applications, teeth whitening, face skin enhancement, red eye removal or eye colorization, and so on. 

EasyPortrait dataset size is **???** and it contains **20 000** RGB images (~17.5K FullHD images) with high quality annotated masks. This dataset is divided into training set, validation set and test set by subject `user_id`. The training set includes 14000 images, the validation set includes 2000 images, and the test set includes 4000 images.

## Downloads
We split dataset of images into 3 archives. Download and unzip them from the following links:

### Links
Click to start downloading:

| Link                                  | Size   |
|---------------------------------------|--------|
| [`train`](https://sc.link/gvlr)       | 18.14G |
| [`test`](https://sc.link/vOEn)        | 5.3G   |
| [`val`](https://sc.link/rKzL)         | 2.6G   |
| [`annotations`](https://sc.link/wPGX) | 235M   |

### Structure
```
segmentation
├── train.zip          # Train set: 14k
├── val.zip            # Validation set: 2k
├── test.zip           # Test set: 4k
├── annotations.zip
│   ├── meta.zip       # Meta-information (width, height, brightness, imhash, user_id)
│   ├── train/     
│   ├── val/       
│   ├── test/      
...
```

## Models
We provide some pre-trained models as the baseline for portrait segmentation and face parsing.

| Model Name                | Metric |
|---------------------------|--------|
| [model_name](https://...) | ...    |
| [model_name](https://...) | ...    |

## Demo
 ```bash
python demo.py -p <PATH_TO_MODEL>
```

## Annotations

Annotations are presented as 2D-images (`png` format) with several classes:

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

Also we provide some additional meta-information for dataset in `annotations/meta.zip` file

|    | attachment_id | user_id | data_hash | width | height | brightness | train | test  | valid |
|---:|:--------------|:--------|:----------|------:|-------:|-----------:|:------|:------|:------|
|  0 | de81cc1c-...  | 1b...   | e8f...    |  1440 |   1920 |        136 | True  | False | False |
|  1 | 3c0cec5a-...  | 64...   | df5...    |  1440 |   1920 |        148 | False | False | True  |
|  2 | d17ca986-...  | cf...   | a69...    |  1920 |   1080 |        140 | False | True  | False |

where:
- `attachment_id` is the image file name without extension
- `user_id' is the unique anonymized user ID
- `data_hash` is the image hash by Perceptual hashing
- `width` is the image width
- `height` is the image height
- `brightness` is the image brightness
- `train`, `test`, `valid` are the binary columns for train / test / val subsets respectively 

### Authors and Credits
- [Alexander Kapitanov](https://www.linkedin.com/in/hukenovs)
- [Karina Kvanchiani](https://www.linkedin.com/in/kvanchiani)
- [Sofia Kirillova](https://www.linkedin.com/in/gofixyourself/)

### Citation
You can cite the paper using the following BibTeX entry:
    @article{EasyPortrait,
        title={EasyPortrait - Face Parsing and Portrait Segmentation Dataset},
        author={Kapitanov, Alexander and Kvanchiani, Karina and Kirillova Sofia},
        journal={arXiv preprint <link>},
        year={2023}
    }

### License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf).
