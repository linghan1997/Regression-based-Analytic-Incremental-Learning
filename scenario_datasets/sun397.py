import os

from .utils import DatasetBase, read_split

from .oxford_pets import OxfordPets

template = ['a photo of a {}.']

print('preparing SUN397 dataset')


class SUN397(DatasetBase):
    dataset_dir = 'sun397'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_SUN397.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # for datum in train:
        #     datum.update_classname(datum.classname + ' scene')
        #
        # for datum in test:
        #     datum.update_classname(datum.classname + ' scene')

        super().__init__(train_x=train, val=val, test=test)
