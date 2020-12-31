from fastai.vision.all import *
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
import json
from os import listdir
from os.path import isfile, join
from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train'
files = [f for f in listdir(TRAIN_PATH) if isfile(join(TRAIN_PATH, f))]


def get_x(r):
    return (Path(TRAIN_PATH) / r["image_id"])


def get_y(r):
    return (r["label"])


with open(DATA_PATH + '/label_num_to_disease_map.json') as f:
    map_label = json.load(f)

df = pd.read_csv(DATA_PATH + '/train.csv', dtype={"label": "str"})
df = df.loc[df['image_id'].isin(files)]
df["label"].replace(map_label, inplace=True)
df.head()
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_x=get_x,
                   get_y=get_y,
                   item_tfms=RandomResizedCrop(460, min_scale=.3),
                   batch_tfms=aug_transforms(size=224, min_scale=.75),
                   )

dsets = dblock.dataloaders(df, num_workers=0)

learn = cnn_learner(dsets, resnet34, metrics=error_rate)  ## fastai automatically freeze the layers
learn.fit(1)
print('12 ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
preds,y,losses = learn.get_preds(with_loss=True)
# def __init__(self, dl, inputs, preds, targs, decoded, losses):
print(preds)
print(y)
print(losses)

interp = ClassificationInterpretation(learn, preds, y, losses, preds, losses)
interp.plot_confusion_matrix()
print('13 ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))






