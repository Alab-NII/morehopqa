"""
Load dataset based on 2wikihop dataset.
"""
from datasets.abstract_dataset_loader import DatasetLoader
import json
import random
random.seed(42)

class MorehopqaLoader(DatasetLoader):
    path = "datasets/files/morehopqa_final.json"

    def __init__(self):
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.length = len(self.data)

    def items(self):
        for item in self.data:
            yield item


class Morehopqa150Loader(DatasetLoader):
    path = "datasets/files/morehopqa_final_150samples.json"

    def __init__(self):
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self.length = len(self.data)

    def items(self):
        for item in self.data:
            yield item