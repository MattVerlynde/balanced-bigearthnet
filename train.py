import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.ShortCNN_RGB import Model as ShortCNNrgb
from models.ShortCNN_All import Model as ShortCNNall
from models.InceptionV3 import Model as InceptionV3
from BigEarthNet import BigEarthNetDataLoader

import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import multilabel_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
TB_PATH = "./tensorboard_logs"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(yhat,y,threshold=0.5):
    yhat = F.sigmoid(yhat)
    # y encode les indexes, s'assurer de la bonne taille de tenseur
    N,C = yhat.shape
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    accuracy = (yhat == y).sum() / (N*C) * 100
    # TP = ((yhat == 1) & (y == 1)).sum()
    # P = y.sum()
    # recall = TP / P
    # print(f"Recall: {recall}")
    return accuracy

def test_metrics(yhat, y, threshold=0.5):
    yhat = F.sigmoid(yhat)
    yhat[yhat >= threshold] = 1
    yhat[yhat < threshold] = 0
    yhat_np = yhat.cpu().numpy()
    y_np = y.cpu().numpy()

    confusion = multilabel_confusion_matrix(y_true = y_np, y_pred = yhat_np)

    return confusion

def confusion_matrix_to_metrics(TP, TN, FP, FN):
    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
        f2 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        f2 = 5 * (precision * recall) / (4 * precision + recall)
    return precision, recall, f1, f2

def test_scores(confusion_matrix):
    # get precision, recall, f1-score, f2-score
    # precision = TP / (TP + FP)
    index = ['precision', 'recall', 'f1', 'f2']
    scores_per_class = pd.DataFrame(columns=index)
    for i in range(len(confusion_matrix)):
        TP = confusion_matrix[i, 1, 1]
        TN = confusion_matrix[i, 0, 0]
        FP = confusion_matrix[i, 0, 1]
        FN = confusion_matrix[i, 1, 0]
        precision, recall, f1, f2 = confusion_matrix_to_metrics(TP, TN, FP, FN)
        row = pd.Series({'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2})
        # scores = scores.append(row, ignore_index=True)
        scores_per_class = pd.concat([scores_per_class, row.to_frame().T], axis=0, ignore_index=True)
    print(scores_per_class)

    scores = pd.DataFrame()
    scores = scores_per_class[['precision', 'recall', 'f1', 'f2']].mean()
    print(scores)
    #add "_macro" to the key
    scores.index = ['macro_' + i for i in scores.index]
    
    conf_matrix = np.sum(confusion_matrix, axis=0)
    TN = conf_matrix[0,0]
    TP = conf_matrix[1,1]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    precision, recall, f1, f2 = confusion_matrix_to_metrics(TP, TN, FP, FN)
    row = pd.Series({'micro_precision': precision, 'recall': recall, 'micro_f1': f1, 'micro_f2': f2})
    scores = pd.concat([scores.to_frame().T, row.to_frame().T], axis=1)
    return scores

def save_scores(scores, DIR_OUTPUT):
    scores.to_csv(f"{DIR_OUTPUT}/scores.csv")


def train(seed, model, epochs, train_loader, val_loader, optim=torch.optim.Adam, lr=1e-3, loss=nn.CrossEntropyLoss(), step_validation=1, storage_path="./", threshold=0.5, verbose=False):
    torch.manual_seed(seed)
    writer = SummaryWriter(f"{storage_path}/{model.name}")
    optimizer = optim(model.parameters(),lr=lr)     # choix optimizer
    model = model.to(device)
    print(f"Running {model.name}")                          # choix loss
    for epoch in range(epochs):
        cumloss, cumacc, count = 0, 0, 0
        model.train()
        # print("-"*10)
        # print(f"Epoch {epoch}")
        for x,y in train_loader:                            # boucle sur les batchs
            optimizer.zero_grad()
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)      # y doit être un tensor (pas un int)
            yhat = model(x)
            l = loss(yhat,y)
            l.backward()
            optimizer.step()
            cumloss += l*len(x)                             # attention, il peut y avoir un batch + petit (le dernier)
            cumacc += accuracy(yhat,y, threshold)*len(x)
            count += len(x)
        writer.add_scalar('loss/train',cumloss/count,epoch)
        writer.add_scalar('accuracy/train',cumacc/count,epoch)
        if epoch % step_validation == 0:
            print("-"*10)
            print(f"Validation {epoch // step_validation}")
            model.eval()
            with torch.no_grad():
                cumloss, cumacc, count = 0, 0, 0
                range_val = val_loader
                if verbose:
                    range_val = tqdm(val_loader)
                for x,y in range_val:
                    x = torch.tensor(x, dtype=torch.float32).to(device)
                    y = torch.tensor(y, dtype=torch.float32).to(device)
                    # x = x.to(device, dtype=torch.float32)
                    # y = y.to(device, dtype=torch.float32)
                    yhat = model(x)
                    cumloss += loss(yhat,y)*len(x)
                    cumacc += accuracy(yhat,y,threshold)*len(x)
                    count += len(x)
                writer.add_scalar('loss/val',cumloss/count,epoch)
                writer.add_scalar('accuracy/val',cumacc/count,epoch)
                print(f"Validation loss = {cumloss/count}")
                print(f"Validation accuracy = {cumacc/count}")

def test(model, test_loader, loss=nn.CrossEntropyLoss(), threshold=0.5, DIR_OUTPUT="./", label_type = 'original'):
    model.eval()
    with torch.no_grad():
        cumloss, cumacc, count = 0, 0, 0
        num_classes = 43 if label_type == 'original' else 19
        conf_matrix = np.zeros((num_classes, 2, 2))
        print("-"*10)
        print("Test")
        TP, TN, FP, FN = np.zeros(43), np.zeros(43), np.zeros(43), np.zeros(43)
        for x,y in test_loader:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            yhat = model(x)
            cumloss += loss(yhat,y)*len(x)
            cumacc += accuracy(yhat,y, threshold=0.5)
            conf_matrix_i = test_metrics(yhat, y, threshold=0.5)
            conf_matrix += conf_matrix_i
            count += len(x)
        print(f"Test loss: {cumloss/count}")
        print(f"Test accuracy: {cumacc/count}")
        # print(f"Test confusion matrix:")
        # print(conf_matrix)
        scores = test_scores(conf_matrix)
        print(f"Test scores:")
        print(scores)
        save_scores(scores, DIR_OUTPUT)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', type=str, required=True)
    parser.add_argument('--model', type=str, default='ShortCNN')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='CrossEntropy')
    parser.add_argument('--batch', type=int, default=500)
    # parser.add_argument('--count', action='store_true')
    # parser.add_argument('--no-count', dest='count', action='store_false')
    parser.add_argument('--rgb', action='store_true')
    # parser.add_argument('--no-rgb', dest='rgb', action='store_false')
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--storage_path', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    # parser.set_defaults(count=False)
    parser.set_defaults(rgb=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    import json
    with open(args.sets) as f:
        sets = json.load(f)
    label_type = sets['label_type']

    # sets = {
    #     'train': '/media/HDD/BigEarthNet-S2-v1.0/records/train.tfrecord',
    #     'val': '/home/verlyndem/Documents/data/val.tfrecord',
    #     'test': '/home/verlyndem/Documents/data/test.tfrecord'
    # }
    # label_type = 'original'

    # sets = {
    #     'train': '/home/verlyndem/Documents/data/small/train_small0_w.tfrecord',
    #     'val': '/home/verlyndem/Documents/data/val.tfrecord',
    #     'test': '/home/verlyndem/Documents/data/test.tfrecord'
    # }
    # label_type = 'original'

    # sets = {
    #     'train': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/train.tfrecord',
    #     'val': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/validation.tfrecord',
    #     'test': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/test.tfrecord'
    # }
    # label_type = 'v2'

    # sets = {
    #     'train': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/strat/train.tfrecord',
    #     'val': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/strat/validation.tfrecord',
    #     'test': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/strat/test.tfrecord'
    # }
    # label_type = 'v2'

    # sets = {
    #     'train': '/media/HDD/BigEarthNet-S2-v2.0/strat/train.tfrecord',
    #     'val': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/validation.tfrecord',
    #     'test': '/media/HDD/BigEarthNet-S2-v2.0/tfrecords/test.tfrecord'
    # }
    # label_type = 'v2'

    # sets = {
    #     'train': '/home/verlyndem/Documents/data/strat/train_strat.tfrecord',
    #     'val': '/home/verlyndem/Documents/data/strat/val_strat.tfrecord',
    #     'test': '/home/verlyndem/Documents/data/strat/test_strat.tfrecord'
    # }
    # label_type = 'original'

    preproc = None
    if args.model == 'ShortCNN' and args.rgb:
        model = ShortCNNrgb(label_type=label_type)
    elif args.model == 'ShortCNN' and not args.rgb:
        model = ShortCNNall(label_type=label_type)
    elif args.model == 'InceptionV3':
        if not args.rgb:
            warnings.warn("InceptionV3 only supports RGB images, rgb flag will be ignored")
        args.rgb = True
        weights = 'IMAGENET1K_V1'
        finetune=False
        if args.finetune:
            finetune=True
            if args.finetune == 2:
                weights = None
        if label_type == 'original':
            num_classes = 43
        else:
            num_classes = 19
        model = InceptionV3(num_classes = num_classes, weights='IMAGENET1K_V1', finetune=True)
        
    else:
        print("Model not found, use ShortCNN or InceptionV3")
        exit()

    if args.optim == 'Adam':
        optim = torch.optim.Adam
    elif args.optim == 'SGD':
        optim = torch.optim.SGD
    else:
        warnings.warn("Optimizer not found, using Adam")
        optim = torch.optim.Adam

    if args.loss == 'CrossEntropy':
        loss = nn.CrossEntropyLoss
    elif args.loss == 'MSE':
        loss = nn.MSELoss
    elif args.loss == 'BCEWithLogits':
        loss = nn.BCEWithLogitsLoss
    else:
        warnings.warn("Loss not found, using BCEWithLogits")
        loss = nn.BCEWithLogitsLoss

    print(f"Number of trainable parameters: {count_parameters(model)}")

    train_loader = BigEarthNetDataLoader(sets['train'], batch_size=args.batch, RGB=args.rgb, shuffle=True, seed=args.seed, label_type=label_type)
    val_loader = BigEarthNetDataLoader(sets['val'], batch_size=args.batch, RGB=args.rgb, seed=args.seed, label_type=label_type)
    test_loader = BigEarthNetDataLoader(sets['test'], batch_size=args.batch, RGB=args.rgb, seed=args.seed, label_type=label_type)

    from codecarbon import OfflineEmissionsTracker
    
    DIR_CARBON = os.path.join(args.storage_path,"codecarbon")
    DIR_OUTPUT = os.path.join(args.storage_path,"output")
    os.makedirs(DIR_CARBON, exist_ok=True)
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=DIR_CARBON, output_file="emissions.csv", log_level="warning")
    tracker.start()
    
    train(seed = args.seed,
        model = model, 
        epochs = args.epochs, 
        train_loader = train_loader, 
        val_loader = val_loader,
        optim = torch.optim.Adam, 
        lr = args.lr,
        loss=loss(),
        storage_path=args.storage_path, 
        threshold=0.5,
        step_validation=args.epochs//5,
        verbose=args.verbose)
    
    tracker.stop()
    
    test(model = model, 
        test_loader = test_loader,
        loss=loss(),
        label_type = label_type,
        threshold=0.5,
        DIR_OUTPUT=DIR_OUTPUT)