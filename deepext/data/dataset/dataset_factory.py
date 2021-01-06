from abc import ABCMeta, abstractmethod
from typing import Tuple, Generator, List
import numpy as np

from torch.utils.data import Dataset


class MultipleDatasetFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_train_test(self, train_indices: np.ndarray, test_indices: np.ndarray) -> Tuple[Dataset, Dataset]:
        pass

    @abstractmethod
    def create_kfold_generator(self, k_indices_ls: List[np.ndarray]) -> Generator[Tuple[Dataset, Dataset]]:
        pass
