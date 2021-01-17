from typing import List, Tuple, Generator
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from .common import create_filepath_ls
from .dataset_factory import MultipleDatasetFactory


class IndexImageMultiDatasetFactory(MultipleDatasetFactory):
    def __init__(self, image_dir_path: str,
                 index_image_dir_path: str, train_transforms, test_transforms,
                 valid_suffixes: List[str] = None):
        self._img_dir_path = image_dir_path
        self._index_img_dir_path = index_image_dir_path
        self._train_transforms, self._test_transforms = train_transforms, test_transforms
        self._valid_suffixes = valid_suffixes

    def create_train_test(self, train_indices: np.ndarray, test_indices: np.ndarray) -> Tuple[Dataset, Dataset]:
        image_path_ls = create_filepath_ls(self._img_dir_path, self._valid_suffixes)
        train_image_path_ls, test_image_path_ls = [], []
        for idx in train_indices:
            train_image_path_ls.append(image_path_ls[idx])
        for idx in test_indices:
            test_image_path_ls.append(image_path_ls[idx])
        train_dataset = IndexImageDataset(image_filename_ls=train_image_path_ls,
                                          image_dir=self._img_dir_path,
                                          index_image_dir=self._index_img_dir_path,
                                          transform=self._train_transforms)
        test_dataset = IndexImageDataset(image_filename_ls=test_image_path_ls,
                                         image_dir=self._img_dir_path,
                                         index_image_dir=self._index_img_dir_path,
                                         transform=self._test_transforms)
        return train_dataset, test_dataset

    def create_kfold_generator(self, k_indices_ls: List[np.ndarray]) -> Generator[Tuple[Dataset, Dataset], None, None]:
        assert len(k_indices_ls) >= 2
        for i in range(len(k_indices_ls) - 1):
            train_indices, test_indices = self.split_kfold_train_test_indices(k_indices_ls, i)
            train_dataset, test_dataset = self.create_train_test(train_indices, test_indices)
            yield train_dataset, test_dataset


class IndexImageDataset(Dataset):
    @staticmethod
    def create(image_dir_path: str, index_image_dir_path: str, transforms, valid_suffixes: List[str] = None):
        image_path_ls = create_filepath_ls(image_dir_path, valid_suffixes)
        return IndexImageDataset(image_path_ls, image_dir_path, index_image_dir_path, transforms)

    def __init__(self, image_filename_ls: List[str], image_dir: str, index_image_dir: str,
                 transform):
        self._transform = transform
        self._image_dir = Path(image_dir)
        self._index_image_dir = Path(index_image_dir)
        self._image_filename_ls = image_filename_ls

    def __len__(self):
        return len(self._image_filename_ls)

    def __getitem__(self, idx: int):
        image_name = self._image_filename_ls[idx]
        image_path = self._image_dir.joinpath(image_name)
        index_image_path = self._index_image_dir.joinpath(f"{Path(image_name).stem}.png")
        image = Image.open(str(image_path)).convert("RGB")
        index_image = Image.open(str(index_image_path))
        return self._transform(image, index_image)
