from typing import Tuple, List
import pickle
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold


# TODO うまくSubsetを用いた仕組みを思いつくまで保留
class TrainTestSplitter:
    @staticmethod
    def create(dataset: Dataset, test_ratio: float) -> 'TrainTestSplitter':
        data_len = len(dataset)
        test_num = int(data_len * test_ratio)
        indices = np.random.permutation(data_len)
        test_indices = indices[:test_num]
        train_indices = indices[test_num:]
        return TrainTestSplitter(dataset, train_indices, test_indices)

    @staticmethod
    def create_from_indices_file(dataset: Dataset, filepath: str):
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            if not isinstance(data, tuple) and len(data) != 2:
                raise RuntimeError(f"Indices pickle file must be 2 tuple object.  {type(data)}")
            train_indices, test_indices = data
            return TrainTestSplitter(dataset, train_indices, test_indices)

    def __init__(self, dataset: Dataset, train_indices: np.ndarray, test_indices: np.ndarray):
        assert train_indices.ndim == 1 and test_indices == 1
        self._dataset = dataset
        self._test_indices = test_indices
        self._train_indices = train_indices

    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._train_indices, self._test_indices

    def save_indices(self, save_filepath: str):
        with open(save_filepath, "wb") as file:
            pickle.dump((self._train_indices, self._test_indices), file)

    def train_test_dataset(self):
        return Subset(dataset=self._dataset, indices=self._train_indices), Subset(dataset=self._dataset,
                                                                                  indices=self._test_indices)


class KFoldSplitter:
    def __init__(self, k: int):
        k_fold = KFold(k)
