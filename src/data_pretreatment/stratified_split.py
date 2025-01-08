#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script creates balanced splits with TFRecord files from BigEarthNet 
# image patches based on csv files that contain patch names.
# 
# stratified_split.py --help can be used to learn how to use this script.
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 8 jan 2024
# Version: 1.0.0
# Usage: stratified_split.py [-h] [-d DATA_FOLDER] [-k NUMBER OF SPLITS] [-o OUTPUT_FOLDER] [-r ROOT_FOLDER] [-tf FLAG TO CREATE TFRECORD FILES]
#                       

import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import json
import argparse
import os
from tensorflow_utils import prep_tf_record_files

def split_sample(data_path, k_splits, output_path):
    data = pd.read_csv(data_path, index_col=0)
    X = data['patch']
    y = data.loc[:, data.columns != 'patch']

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    mskf = MultilabelStratifiedKFold(k_splits=k_splits, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in mskf.split(X, y):
        X_fold = X[test_index]
        print(X_fold)
        pd.Series(X_fold).to_csv(f"{output_path}/fold_{i}.csv", index=False, header=False)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        'This script creates stratified splits for the BigEarthNet dataset.')
    parser.add_argument('-d', '--data_path', dest = 'data_path',
                        help = 'path to the csv file containing the data')
    parser.add_argument('-k', '--k_splits', dest = 'k_splits', type=int,
                        help = 'number of splits to create')
    parser.add_argument('-o', '--output_path', dest = 'output_path',
                        help = 'path to the folder where the splits will be saved')
    parser.add_argument('-r', '--root_folder', dest = 'root_folder',
                        help = 'root folder path contains multiple patch folders')
    
    #flag to create tfrecord files, if the flag then yes, if no flag then no, default is no
    parser.add_argument('-tf', '--tfrecord', dest = 'tfrecord', default=False, action='store_true',
                        help = 'flag to create tfrecord files')
    
    args = parser.parse_args()

    split_sample(args.data_path, args.k_splits, args.output_path)

    if args.tfrecord:
        GDAL_EXISTED = False
        RASTERIO_EXISTED = False
        try:
            import gdal
            GDAL_EXISTED = True
            print('INFO: GDAL package will be used to read GeoTIFF files')
        except ImportError:
            try:
                import rasterio
                RASTERIO_EXISTED = True
                print('INFO: rasterio package will be used to read GeoTIFF files')
            except ImportError:
                print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
                exit()

        split_names = []
        patch_names_list = []
        for x in os.listdir(args.output_path):
            if x[-4:] == '.csv':
                split_names.append(x[:-4])
                patch_names_list.append(pd.read_csv(f"{args.output_path}/{x}", index_col=False, header=None).loc[:,0].to_list())

        with open('data_pretreatment/label_indices.json', 'rb') as f:
            label_indices = json.load(f)

        # print(split_names)
        print(patch_names_list)
        prep_tf_record_files(
            root_folder = args.root_folder, 
            out_folder = args.output_path, 
            split_names = split_names, 
            patch_names_list = patch_names_list, 
            label_indices = label_indices, 
            GDAL_EXISTED = GDAL_EXISTED, 
            RASTERIO_EXISTED = RASTERIO_EXISTED)