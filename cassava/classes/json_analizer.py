import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


class JsonAnalizer:

    def __init__(self, path):
        self.path = path
        self.data = self.loadJson()
        self.castKeysToInteger()

    def loadJson(self):
        with open(self.path) as f:
            return json.load(f)

    def castKeysToInteger(self):
        self.data = {int(k): v for k, v in self.data}

    def print(self):
        print(self.data)

    def getLabels(self):
        return self.data
