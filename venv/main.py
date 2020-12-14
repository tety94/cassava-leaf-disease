from fastai.vision.all import *
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
import json


def main():
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms=RandomResizedCrop(460, min_scale=.3),
                       batch_tfms=aug_transforms(size=224, min_scale=.75),
                       )

    dsets = dblock.dataloaders(df, num_workers=0)

    learn = cnn_learner(dsets, resnet34, metrics=error_rate)  ## fastai automatically freeze the layers

    # learn.unfreeze() # if I want to unfreeze the model, I have to do it manually
    learn.lr_find()
    learn.fit_one_cycle(2, 3e-3)
    learn.model

    learn.unfreeze()
    # learn.freeze_to(-2)
    #
    # learn.lr_find(start_lr = 1e-9, end_lr = 1e-5, num_it = 100)
    # learn.unfreeze()
    # learn.freeze()
    #
    # learn.lr_find(start_lr = 1e-9, end_lr = 1e-5, num_it = 100)
	
	return learn


def get_x(r):
    return (Path(TRAIN_PATH) / r["image_id"])


def get_y(r):
    return (r["label"])


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = dir_path + '/data/'
    TRAIN_PATH = DATA_PATH + '/train'
    list(Path(TRAIN_PATH).glob("*"))[:5]

    img_path = Path(TRAIN_PATH).glob("*")
    del img_path

    with open(DATA_PATH + '/label_num_to_disease_map.json') as f:
        map_label = json.load(f)
    map_label
    df = pd.read_csv(DATA_PATH + '/train.csv', dtype={"label": "str"})

    df["label"].replace(map_label, inplace=True)

    df.head()

    l = main()
	
	interp = ClassificationInterpretation.from_learner(l)
	interp.plot_confusion_matrix()

	interp.plot_top_losses(3, n_rows = 1)
