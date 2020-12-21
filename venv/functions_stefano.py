import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from os import listdir
from os.path import isfile, join


class JsonAnalizer:

    def __init__(self, path):
        self.path = path
        self.data = self.loadJson()

    def loadJson(self):
        with open(self.path) as f:
            return json.load(f)

    def print(self):
        print(self.data)

    def getLabels(self):
        return self.data


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


def plotImagesByPaths(imagesNames, basePath, offset, nrows, ncols):
    lenght = len(imagesNames)
    cycles = lenght / (nrows * ncols)
    firstCont = 0
    while firstCont < cycles:
        figure, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * ncols), constrained_layout=True)
        cont = 0
        for i in imagesNames[offset * nrows * ncols: (offset + 1) * nrows * ncols ]:
            im = img.imread(basePath + i)
            ax.ravel()[cont].imshow(im)
            ax.ravel()[cont].set_title(i)
            ax.ravel()[cont].set_axis_off()
            cont = cont + 1
        plt.show()
        firstCont = firstCont + 1
        offset = offset + 1
