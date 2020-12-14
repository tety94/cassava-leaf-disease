import json
import pandas as pd


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
        self.data = pd.read_csv(self.path, dtype={"label": "str"})

    def read(self):
        return pd.read_csv(self.path, dtype={"label": "str"})

    def printStats(self, jsonLabel):
        self.data["label"].replace(jsonLabel, inplace=True)
        print(self.data.groupby('label').count())


