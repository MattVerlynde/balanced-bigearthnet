import pandas as pd
import numpy as np
import os
import argparse

def separate_csv(path):
    df = pd.read_csv(path, index_col=None)
    path_output = path[:-25]
    df_train = df.iloc[np.where(df["sample"]=="train")]
    df_val = df.iloc[np.where(df["sample"]=="val")]
    df_train = df_train.reset_index()
    df_train = df_train.drop(columns=['index', 'Unnamed: 0'])
    j=1
    for i in range(len(df_val)):
        df_val.iloc[i,0] = df_val.iloc[i,0] - j
        j+=1
    df_val = df_val.set_index("Unnamed: 0")
    df_train.index.names = ['epoch']
    df_val.index.names = ['epoch']
    df_train.to_csv(path_output + "train.csv")
    df_val.to_csv(path_output + "val.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate csv file into train and val csv files')
    parser.add_argument('--path', type=str, help='Path to the csv file')
    args = parser.parse_args()
    separate_csv(args.path)
    #separate_csv("results/conso-classif-deep/run_13/group_0/output/losses_accuracies_all.csv")