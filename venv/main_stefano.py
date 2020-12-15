from functions_stefano import *
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path2 = os.getcwd()

DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train'
jsonPath = DATA_PATH + '/label_num_to_disease_map.json'
csvPath = DATA_PATH + '/train.csv'

jsAnalyzer = JsonAnalizer(jsonPath)
# jsAnalyzer.print()

df = pandasUtils(csvPath)
df.printStats(jsAnalyzer.getLabels())


imagesList = [f for f in listdir(TRAIN_PATH) if isfile(join(TRAIN_PATH, f))]
df.takeExistingImages(imagesList) #now the dataset has only existing images
df.printInfo()


def getPercentageLabels(df):
    d = {}
    for label in jsAnalyzer.getLabels().keys():
        d[(label, jsAnalyzer.getLabels()[label])] = round(100 * df[df.label == label].shape[0] / df.shape[0], 2)
    return d


# print(getPercentageLabels(df.data))