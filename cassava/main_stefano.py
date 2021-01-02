from cassava.classes.json_utilities import JsonUtilities
from cassava.classes.pandas_utilities import pandasUtilities
from cassava import functions_stefano
import os
from pathlib import Path
from cassava.functions_stefano import *

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train/'
TRAIN_CROPPED_PATH = DATA_PATH + '/train_crop/'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

jsAnalyzer = JsonUtilities(jsonPath)
# jsAnalyzer.print()

df = pandasUtilities(csvPath)
# df.takeExistingImages(TRAIN_PATH)  # now the dataset has only existing images
df.mapLabels(jsAnalyzer.getLabels())
df.printStats()

labels = list(jsAnalyzer.data.keys())
imagesPath = df.getImagesNamesByLabels(labels)

cropImage(image='1.jpg', basePath=TRAIN_PATH, savePath=TRAIN_CROPPED_PATH,
          new_width=500, new_height=500, save=True, show= False)
# plotImagesByPaths(imagesNames=imagesPath, basePath=TRAIN_PATH, offset=0, nrows=3, ncols=4)

# cropAllImages(imagesList = imagesPath, basePath = TRAIN_PATH, savePath = TRAIN_CROPPED_PATH, show=True, save=False)
