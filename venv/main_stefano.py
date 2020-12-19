from functions_stefano import *
import os
from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train/'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

jsAnalyzer = JsonAnalizer(jsonPath)
# jsAnalyzer.print()

df = pandasUtils(csvPath)
df.takeExistingImages(TRAIN_PATH)  # now the dataset has only existing images
# df.printInfo()

labels = [4]
imagesPath = df.getImagesNamesByLabels(labels)


plotImagesByPaths(imagesNames=imagesPath, basePath=TRAIN_PATH, offset=36, nrows=3, ncols=4)

# def get_x(r):
#     # print( r)
#     return TRAIN_PATH
#     return (Path(TRAIN_PATH) / r["image_id"])
#
#
# def get_y(r):
#     return (r["label"])
#
# fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
#                    get_items=get_x,
#                    get_y=get_y,
#                    splitter=RandomSplitter(valid_pct=0.2, seed=42),
#                    item_tfms=RandomResizedCrop(224, min_scale=0.5),
#                    batch_tfms=aug_transforms())
#
#
#
# dls = fields.dataloaders(TRAIN_PATH)
