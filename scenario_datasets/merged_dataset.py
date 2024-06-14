import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

import numpy as np


class MergedDataset(DatasetBase):
    def __init__(self):
        train, val, test = [], [], []
        super().__init__(train_x=train, val=val, test=test)

    def merge(self, new_dataset):
        combined_dataset = [*new_dataset.train_x, *new_dataset.val, *new_dataset.test]
        for item in combined_dataset:
            item.update_label(self.num_classes)

        self.merge_dataset(new_dataset)
        self.update_num_classes(self.train_x)

    def shape(self):
        return [len(self.train_x), len(self.val), len(self.test)]