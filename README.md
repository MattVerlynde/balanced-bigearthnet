# Implementation on PyTorch for deep learning models on the BigEarthNet dataset

This repository contains the code for to train the BigEarthNet dataset [[1]](#1), using machine learning algorithms.

## Repository structure

``
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
├── performance-tracking
│   ├── experiments
│   │   ├── conso
│   │   │   ├── analyse_stats.py
│   │   │   ├── get_conso.py
│   │   │   ├── get_stats.py
│   │   │   ├── query_influx.sh
│   │   │   ├── simulation_metrics_exec.sh
│   │   │   ├── stats_summary_deep.py
│   │   │   └── stats_summary.py
│   │   ├── conso_change
│   │   │   ├── cd_sklearn_pair_var.py
│   │   │   ├── change-detection.py
│   │   │   ├── functions.py
│   │   │   ├── get_perf.py
│   │   │   ├── helpers
│   │   │   │   └── multivariate_images_tool.py
│   │   │   ├── main.py
│   │   │   ├── param_change_glrt_2images.yaml
│   │   │   ├── param_change_interm.yaml
│   │   │   ├── param_change_logdiff_2images.yaml
│   │   │   ├── param_change_robust_2images.yaml
│   │   │   ├── param_change_robust.yaml
│   │   │   └── param_change.yaml
│   │   ├── conso_classif_deep
│   │   │   ├── classif_deep.py
│   │   │   ├── get_perf.py
│   │   │   ├── get_scores.py
│   │   │   ├── param_classif_deep_Inception.yaml
│   │   │   ├── param_classif_deep_SCNN_10.yaml
│   │   │   ├── param_classif_deep_SCNN_strat.yaml
│   │   │   ├── param_classif_deep_SCNN.yaml
│   │   │   ├── param_classif_deep_test.yaml
│   │   │   ├── param_classif_deep.yaml
│   │   │   ├── read_event.py
│   │   │   ├── read_events.py
│   │   │   └── simulation_metrics_exec.sh
│   │   └── conso_clustering
│   │       ├── clustering.py
│   │       ├── get_perf.py
│   │       ├── helpers
│   │       │   └── processing_helpers.py
│   │       ├── param_clustering_interm.yaml
│   │       ├── param_clustering.yaml
│   │       ├── plot_clustering.py
│   │       └── utils_clustering.py
│   ├── plot_usage.py
│   ├── README.md
│   └── simulation_metrics_exec.sh
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
│       ├── __pycache__
│       │   ├── InceptionV3.cpython-38.pyc
│       │   ├── ShortCNN_All.cpython-38.pyc
│       │   ├── ShortCNN.cpython-38.pyc
│       │   └── ShortCNN_RGB.cpython-38.pyc
│       ├── ShortCNN_All.py
│       └── ShortCNN_RGB.py
└── train.py
``

## Usage

``python train.py --sets [JSON PATH WITH SET PATHS] --epochs [NUMBER OF EPOCHS] --optim [OPTIIMIZER USED] --lr [FLOAT LEARNING RATE] --loss [LOSS FUNCTION USED] --batch [BATCH SIZE] --finetune [FINETUNING LEVEL] --seed [RANDOM SEED] --storage_path [EVENT STORAGE PATH] --count --rgb``

``stratified_split.py [-h] [-d DATA_FOLDER] [-k NUMBER OF SPLITS] [-o OUTPUT_FOLDER] [-r ROOT_FOLDER] [-tf FLAG TO CREATE TFRECORD FILES]``

``prep_splits.py [-h] [-r ROOT_FOLDER] [-o OUT_FOLDER] [-n PATCH_NAMES [PATCH_NAMES ...]]``

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

## References
<a id="1">[1]</a> 
Gencer Sumbul, Arne de Wall, Tristan Kreuziger, Filipe
Marcelino, Hugo Costa, Pedro Benevides, Mario Cae-
tano, Begum Demir, and Volker Markl, “Bigearthnet-
mm: A large-scale, multimodal, multilabel benchmark
archive for remote sensing image classification and re-
trieval,” IEEE Geoscience and Remote Sensing Maga-
zine, vol. 9, no. 3, pp. 174–180, Sept. 2021.


