from functions_stefano import *
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train/'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

jsAnalyzer = JsonAnalizer(jsonPath)
# jsAnalyzer.print()

df = pandasUtils(csvPath)
# df.printStats(jsAnalyzer.getLabels())

imagesList = [f for f in listdir(TRAIN_PATH) if isfile(join(TRAIN_PATH, f))]
df.takeExistingImages(imagesList)  # now the dataset has only existing images
# df.printInfo()

labels = [1]
imagesPath = df.getImagesNamesByLabels(labels)

plotImagesByPaths(imagesNames= imagesPath, basePath= TRAIN_PATH, offset = 0, nrows = 3, ncols = 4)

