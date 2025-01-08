# Implementation on PyTorch for deep learning models on the BigEarthNet dataset

This repository contains the code for to train the BigEarthNet dataset [[1]](#1), using machine learning algorithms.

## Repository structure



## Usage

``python train.py --sets [JSON PATH WITH SET PATHS] --epochs [NUMBER OF EPOCHS] --optim [OPTIIMIZER USED] --lr [FLOAT LEARNING RATE] --loss [LOSS FUNCTION USED] --batch [BATCH SIZE] --finetune [FINETUNING LEVEL] --seed [RANDOM SEED] --storage_path [EVENT STORAGE PATH] --count --rgb``

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


