from fastai.vision.all import *
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
import json

import matplotlib.pyplot as plt 
import matplotlib.image as img

dir_path = os.getcwd() 
DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

with open(DATA_PATH + '/label_num_to_disease_map.json') as f:
    map_label = json.load(f)
	
df = pd.read_csv(csvPath, dtype={"label": "str"})

imagesList = [f for f in os.listdir(TRAIN_PATH) if os.path.isfile(os.path.join(TRAIN_PATH, f))]
df = df[df['image_id'].isin(imagesList)]

def getPercentageLabels(df):
    
    d = {}
    for label in map_label.keys():
        d[(label, map_label[label])] = round(100*df[df.label == label].shape[0]/df.shape[0], 2)
    return d
	
getPercentageLabels(df)

def PrintImages(df, label, from_, to_, ncols = 3):
    """
    
    This function plots images contained in DataFrame *df*
    and with a label *label*, from the image *from*
    to the image *to*.
    The parameter *ncols* sets the number of columns of
    the output showed.
    """
    
    df_ = df[df['label'] == label].reset_index(drop = True)
    df_ = df_[from_:to_ + 1]
    nrows = int((to_ - from_)/ncols) + 1
    
    figure, ax = plt.subplots(nrows = nrows, ncols = ncols, 
                              figsize=(10, 3*nrows), constrained_layout=True)
    
    for i in df_.index:
        im = img.imread(TRAIN_PATH + '/' + df_.loc[i, 'image_id'])
        
        idx = i -from_
        ax.ravel()[idx].imshow(im)
        ax.ravel()[idx].set_title("Image_" + str(i))
        ax.ravel()[idx].set_axis_off()
    plt.show()
	
PrintImages(df, '4', 2, 21, 4)

