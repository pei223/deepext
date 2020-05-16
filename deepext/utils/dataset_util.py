from typing import Tuple, List
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from pathlib import Path
from PIL import Image
import numpy as np
import random


class SimpleGANDataSet(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.filepaths = list(Path(root_dir).glob("*.png")) + list(Path(root_dir).glob("*.jpg")) + list(
            Path(root_dir).glob("*.jpeg"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(str(self.filepaths[idx]))
        if self.transform:
            img = self.transform(img)
        return img


def random_subset_dataset(dataset: Dataset, ratio: float):
    data_len = len(dataset)
    subset_count = int(data_len * ratio)
    subset_indices = np.random.permutation(data_len)[:subset_count]
    return Subset(dataset, indices=subset_indices)


class LabelAndDataTransforms:
    def __init__(self, transform_sets=List[Tuple[any, any]]):
        """
        :param transform_sets: list of data and label transforms [(data_transform, label_transform), ...]
        """
        self._transform_sets = transform_sets

    def __call__(self, img, label):
        for i in range(len(self._transform_sets)):
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            data_transform, label_transform = self._transform_sets[i]
            img = data_transform(img) if data_transform is not None else img
            label = label_transform(label) if label_transform is not None else label
        return img, label
