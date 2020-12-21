from typing import Tuple

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, Subset


def random_subset_dataset(dataset: Dataset, ratio: float):
    data_len = len(dataset)
    subset_count = int(data_len * ratio)
    subset_indices = np.random.permutation(data_len)[:subset_count]
    return Subset(dataset, indices=subset_indices)


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir: str, image_transform):
        self._image_transform = image_transform
        image_dir_path = Path(image_dir)
        self._image_file_path_ls = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.jpeg")) + list(
            image_dir_path.glob("*.png")) + list(image_dir_path.glob("*.bmp"))
        self._current_image_size = None
        self._current_file_path = None

    def __len__(self):
        return len(self._image_file_path_ls)

    def current_image_size(self) -> Tuple[int, int]:
        return self._current_image_size

    def current_file_path(self) -> str:
        return str(self._current_file_path)

    def __getitem__(self, idx):
        file_path = self._image_file_path_ls[idx]
        self._current_file_path = file_path
        img = Image.open(str(file_path))
        img = img.convert("RGB")
        self._current_image_size = img.size[0], img.size[1]
        if self._image_transform:
            img = self._image_transform(img)
        return img
