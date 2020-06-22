import numpy as np
from torch.utils.data import Dataset, Subset


def random_subset_dataset(dataset: Dataset, ratio: float):
    data_len = len(dataset)
    subset_count = int(data_len * ratio)
    subset_indices = np.random.permutation(data_len)[:subset_count]
    return Subset(dataset, indices=subset_indices)