from typing import Tuple, List
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TrainTestIndicesManager:
    @staticmethod
    def create_from_indices_file(filepath: str) -> 'TrainTestIndicesManager':
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            if not isinstance(data, tuple) and len(data) != 2:
                raise RuntimeError(f"Indices pickle file must be 2 tuple object.  {type(data)}")
            train_indices, test_indices = data
            return TrainTestIndicesManager(train_indices, test_indices)

    @staticmethod
    def create(data_len: int, test_ratio: float) -> 'TrainTestIndicesManager':
        test_num = int(data_len * test_ratio)
        train_num = data_len - test_num
        indices = np.random.permutation(data_len)
        train_indices = indices[:train_num]
        test_indices = indices[train_indices:]
        return TrainTestIndicesManager(train_indices, test_indices)

    def __init__(self, train_indices: np.ndarray, test_indices: np.ndarray):
        assert train_indices.ndim == 1 and test_indices.ndim == 1
        self._train_indices, self._test_indices = train_indices, test_indices

    def train_test_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._train_indices, self._test_indices

    def save_indices(self, save_filepath: str):
        with open(save_filepath, "wb") as file:
            pickle.dump((self._train_indices, self._test_indices), file)


class KFoldIndicesManager:
    @staticmethod
    def create(data_len: int, k: int) -> 'KFoldIndicesManager':
        if k < 2 or k > 100:
            raise RuntimeError("K must be 0~100")
        all_indices = np.random.permutation(data_len)
        result = []
        block_data_num = int(data_len / k)
        for i in range(k):
            result.append(all_indices[block_data_num * i:block_data_num * i + 1])
        return KFoldIndicesManager(result)

    @staticmethod
    def create_from_indices_file(filepath: str) -> 'KFoldIndicesManager':
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            if not isinstance(data, list):
                raise RuntimeError(f"KFold Indices pickle file must be list object.  {type(data)}")
            return KFoldIndicesManager(data)

    def __init__(self, indices_ls: List[np.ndarray]):
        self._indices_ls = indices_ls


def create_filepath_ls(image_dir_path: str, valid_suffixes: List[str] = None) -> List[str]:
    valid_suffixes = valid_suffixes or ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_path_ls = []
    image_dir = Path(image_dir_path)
    for suffix in valid_suffixes:
        image_path_ls += list(image_dir.glob(suffix))
    image_path_ls.sort()
    return image_path_ls


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
        """
        :return: (width, height)
        """
        return self._current_image_size

    def current_file_path(self) -> str:
        return str(self._current_file_path)

    def __getitem__(self, idx):
        file_path = self._image_file_path_ls[idx]
        self._current_file_path = file_path
        img = Image.open(str(file_path))
        img = img.convert("RGB")
        width, height = img.size[:2]
        self._current_image_size = width, height
        if self._image_transform:
            img = self._image_transform(img)
        return img
