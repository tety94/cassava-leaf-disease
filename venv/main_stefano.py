from functions_stefano import *
import os
from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train/'
TRAIN_CROPPED_PATH = DATA_PATH + '/train_crop/'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

jsAnalyzer = JsonAnalizer(jsonPath)
# jsAnalyzer.print()

df = pandasUtils(csvPath)
df.takeExistingImages(TRAIN_PATH)  # now the dataset has only existing images
# df.printInfo()

labels = [0]
imagesPath = df.getImagesNamesByLabels(labels)

# plotImagesByPaths(imagesNames=imagesPath, basePath=TRAIN_PATH, offset=0, nrows=3, ncols=4)

cropAllImages(imagesList = imagesPath, basePath = TRAIN_PATH, savePath = TRAIN_CROPPED_PATH, show=False)

