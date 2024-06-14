import os

from .utils import DatasetBase, read_split


template = ['a photo of a {}.']

print('preparing Stanford_Cars dataset')

class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.dataset_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)