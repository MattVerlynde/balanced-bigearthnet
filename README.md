# Implementation on PyTorch for deep learning models on the BigEarthNet dataset

This repository contains the code for to train the BigEarthNet dataset [[1]](#1), using machine learning algorithms.

<span style="background-color:rgb(255, 0, 0)">Work in progress</span>

## Repository structure

```bash
.
├── data
│   └── splits
│       ├── original
│       │   ├── test.csv
│       │   ├── train.csv
│       │   └── val.csv
│       └── resampled
│           ├── test.csv
│           ├── train.csv
│           └── val.csv
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
│       ├── InceptionV2.py
│       ├── InceptionV3.py
│       ├── ShortCNN_All.py
│       └── ShortCNN_RGB.py
└── train.py
```

## Download data

Data description and download link are available on this [https://bigearth.net/v1.0.html](link).

<span style="background-color:rgb(255, 145, 0)">Warning:</span> ~66GB are required to store the data

## Usage

```bash
conda env create -f environment.yml
conda activate balanced-bigearthnet
```

```bash
python train.py --sets [JSON PATH WITH SET PATHS] --epochs [NUMBER OF EPOCHS] --optim [OPTIIMIZER USED] --lr [FLOAT LEARNING RATE] --loss [LOSS FUNCTION USED] --batch [BATCH SIZE] --finetune [FINETUNING LEVEL] --seed [RANDOM SEED] --storage_path [EVENT STORAGE PATH] --count --rgb
```

```bash
stratified_split.py [-h] [-d DATA_FOLDER] [-k NUMBER OF SPLITS] [-o OUTPUT_FOLDER] [-r ROOT_FOLDER] [-tf FLAG TO CREATE TFRECORD FILES]```

```bash
prep_splits.py [-h] [-r ROOT_FOLDER] [-o OUT_FOLDER] [-n PATCH_NAMES [PATCH_NAMES ...]]
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


## References
>  <a id="1">[1]</a>  G. Sumbul, M. Charfuelan, B. Demir, V. Markl, “[BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf)”, IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.

## Licence

MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.