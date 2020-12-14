from functions_stefano import *
import os
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = dir_path + '/data/'
TRAIN_PATH = DATA_PATH + '/train'

jsonPath = DATA_PATH + '/label_num_to_disease_map.json'

jsAnalyzer = JsonAnalizer(jsonPath)
jsAnalyzer.print()

csvPath = DATA_PATH + '/train.csv'
df = pandasUtils(csvPath)
df.printStats(jsAnalyzer.getLabels())
