import os

from .utils import DatasetBase, read_split

template = ['{} texture.']

print('preparing DTD dataset')


class DescribableTextures(DatasetBase):

    dataset_dir = 'dtd'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_DescribableTextures.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

