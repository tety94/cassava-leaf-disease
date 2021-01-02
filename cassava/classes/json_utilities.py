import json

class JsonUtilities:

    def __init__(self, path):
        self.path = path
        self.data = self.loadJson()
        self.castKeysToInteger()

    def loadJson(self):
        with open(self.path) as f:
            return json.load(f)

    def castKeysToInteger(self):
        self.data = {int(k): v for k, v in self.data.items()}

    def print(self):
        print(json.dumps(self.data, indent=4))

    def getLabels(self):
        return self.data
