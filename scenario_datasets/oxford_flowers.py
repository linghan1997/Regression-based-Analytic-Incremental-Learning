import os

from .utils import DatasetBase, read_split

template = ['a photo of a {}, a type of flower.']

print('preparing Oxford_Flowers dataset')

class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)