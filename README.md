# Implementation on PyTorch for deep learning models on the BigEarthNet dataset

This repository contains the code for to train deep learning models on the BigEarthNet dataset [[1]](#1).

**WORK IN PROGRESS**

![Label distribution](./doc/split_label_distribution.jpg)

## Repository structure

```bash
.
├── data
│   ├── original
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── resampled
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
├── environment.yml
├── README.md
├── sets.json
├── src
│   ├── BigEarthNet.py
│   ├── data_pretreatment
│   │   ├── label_indices.json
│   │   ├── prep_splits.py
│   │   ├── stratified_split.py
│   │   └── tensorflow_utils.py
│   └── models
│       ├── InceptionV1.py
│       ├── InceptionV3.py
│       ├── ShortCNN_All.py
│       └── ShortCNN_RGB.py
└── train.py
```

## Installation

### Data download
Data description and download link are available on this [https://bigearth.net/v1.0.html](link).

**Warning:** ~66GB are required to store the data

### Dependencies

Python 3.8.19, PyTorch 2.0.1, Tensorflow 2.13.1, iterative-stratification 0.1.9 [[2]](#2)
```bash
conda env create -f environment.yml
conda activate balanced-bigearthnet
```

## Usage

To **create the TFRecord files** containing the data and used in the training file, use the file `prep_splits.py` made by G. Sumbul et al. [[1]](#1)
```bash
prep_splits.py [-h] [-r ROOT_FOLDER] [-o OUT_FOLDER] [-n PATCH_NAMES [PATCH_NAMES ...]]
```
Example:
```bash
prep_splits.py -r BigEarthNet-S2-v1.0/BigEarthNet-v1.0 -o results -n data/original/train.csv data/original/val.csv data/original/test.csv
```

To **create balanced splits** to create the TFRecord files, use the file `stratified_split.py`.
```bash
stratified_split.py [-h] [-d DATA_FILE] [-k NUMBER OF SPLITS] [-o OUTPUT_FOLDER] [-r ROOT_FOLDER] [-tf]
```

To **train your model** using the the TFRecord files, use the file `train.py`.
```bash
train.py [-h] [--sets JSON_PATH_WITH_TFRECORD_PATHS] [--epochs NUMBER_OF_EPOCHS] [--optim OPTIIMIZER_USED] [--lr FLOAT_LEARNING_RATE] [--loss LOSS_FUNCTION] [--batch BATCH_SIZE] [--finetune FINETUNING_LEVEL] [--seed RANDOM_SEED] [--storage_path EVENT_STORAGE_PATH] [--count] [--rgb]
```

To **plot your results** after training using the training event file created, use the file `read_event.py`.
```bash
python src/read_event.py [-h] [--storage_path EVENT_STORAGE_PATH]
```

## Dataset description

| Acronym  | Label |
|----------|-------|
| NIAL     | Non-irrigated arable land |
| MF       | Mixed forest |
| CF       | Coniferous forest |
| TWS      | Transitional woodland/shrub |
| BLF      | Broad-leaved forest |
| LPOASANV | Land principally occupied by agriculture, with significant areas of natural vegetation |
| CCP      | Complex cultivation patterns |
| Pa       | Pastures |
| SO       | Sea and ocean |
| DUF      | Discontinuous urban fabric |
| WB       | Water bodies |
| AFA      | Agro-forestry areas |
| Pe       | Peatbogs |
| PIL      | Permanently irrigated land |
| OG       | Olive groves |
| ICU      | Industrial or commercial units |
| NG       | Natural grassland |
| SV       | Sclerophyllous vegetation |
| CUF      | Continuous urban fabric |
| WC       | Water courses |
| V        | Vineyards |
| SLF      | Sport and leisure facilities |
| ACAPC    | Annual crops associated with permanent crops |
| FTBP     | Fruit trees and berry plantations |
| MES      | Mineral extraction sites |
| IM       | Inland marshes |
| RF       | Rice fields |
| MH       | Moors and heathland |
| RRNAL    | Road and rail networks and associated land |
| BR       | Bare rock |
| BDS      | Beaches, dunes, sands |
| GUA      | Green urban areas |
| SVA      | Sparsely vegetated areas |
| SM       | Salt marshes |
| CS       | Construction sites |
| CL       | Coastal lagoons |
| DS       | Dump sites |
| IF       | Intertidal flats |
| E        | Estuaries |
| A        | Airports |
| PA       | Port areas |
| S        | Salines |
| BA       | Burnt areas |

## Authors

Matthieu Verlynde ([matthieu.verlynde@univ-smb.fr](mailto:matthieu.verlynde@univ-smb.fr)), Ammar Mian ([ammar.mian@univ-smb.fr](mailto:ammar.mian@univ-smb.fr)), Yajing Yan ([yajing.yan@univ-smb.fr](mailto:yajing.yan@univ-smb.fr))

## References
>  <a id="1">[1]</a>  G. Sumbul, M. Charfuelan, B. Demir, V. Markl, “[BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf)”, IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.<br>
>  <a id="2">[2]</a>  C. Sechidis, G. Tsoumakas, I. Vlahavas, “[On the stratification of multi-label data](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10)” Machine Learning and Knowledge Discovery in Databases, D. Gunopulos, T. Hofmann, D. Malerba, and M. Vazirgiannis, Eds., Berlin, Heidelberg, 2011, pp. 145–158, Springer Berlin Heidelberg.