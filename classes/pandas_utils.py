import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


class pandasUtils:

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path, dtype={"label": "int"})

    def read(self):
        return pd.read_csv(self.path, dtype={"label": "int"})

    def printStats(self, jsonLabel):
        self.data["label"].replace(jsonLabel, inplace=True)
        g = self.data['label']
        df = pd.concat([g.value_counts(),
                        g.value_counts(normalize=True).mul(100)], axis=1, keys=('counts', 'percentage'))
        print(df)

    def takeExistingImages(self, path):
        imagesList = [f for f in listdir(path) if isfile(join(path, f))]
        existingBoolean = self.data.image_id.isin(imagesList)  # return True or False
        self.data = self.data[existingBoolean]

    def printInfo(self):
        print(self.data.info())

    def getImagesPathByLabels(self, labels, path):
        parsedDataSet = self.parseDataSetByLabels(labels)
        return [Path(path + i) for i in parsedDataSet['image_id']]

    def parseDataSetByLabels(self, labels):
        return self.data[self.data['label'].isin(labels)]

    def getImagesNamesByLabels(self, labels):
        parsedDataSet = self.parseDataSetByLabels(labels)
        return [i for i in parsedDataSet['image_id']]
