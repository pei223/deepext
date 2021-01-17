from abc import ABCMeta, abstractmethod
from typing import Tuple, Generator, List
import numpy as np

from torch.utils.data import Dataset


class MultipleDatasetFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_train_test(self, train_indices: np.ndarray, test_indices: np.ndarray) -> Tuple[Dataset, Dataset]:
        pass

    @abstractmethod
    def create_kfold_generator(self, k_indices_ls: List[np.ndarray]) -> Generator[Tuple[Dataset, Dataset], None, None]:
        pass

    def split_kfold_train_test_indices(self, k_indices_ls: List[np.ndarray], test_order: int) -> Tuple[
        np.ndarray, np.ndarray]:
        train_indices = np.array([])
        test_indices = None
        for i, indices in enumerate(k_indices_ls):
            if i == test_order:
                test_indices = indices
                continue
            train_indices = np.concatenate([train_indices, indices])
        return train_indices, test_indices
