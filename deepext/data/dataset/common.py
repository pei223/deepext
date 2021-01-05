from typing import Tuple, List

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def create_filepath_ls(image_dir_path: str, valid_suffixes: List[str] = None) -> List[str]:
    valid_suffixes = valid_suffixes or ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_path_ls = []
    image_dir = Path(image_dir_path)
    for suffix in valid_suffixes:
        image_path_ls += list(image_dir.glob(suffix))
    image_path_ls.sort()
    return image_path_ls


def split_train_test_indices(data_len: int, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    test_num = int(data_len * test_ratio)
    train_num = data_len - test_num
    indices = np.random.permutation(data_len)
    train_indices = indices[:train_num]
    test_indices = indices[train_indices:]
    return train_indices, test_indices


def split_k_fold_indices(data_len: int, k: int) -> List[np.ndarray]:
    all_indices = np.random.permutation(data_len)
    result = []
    block_data_num = int(data_len / k)
    for i in range(k):
        result.append(all_indices[block_data_num * i:block_data_num * i + 1])
    return result


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
