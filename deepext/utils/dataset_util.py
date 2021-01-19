from typing import Dict, List, Tuple
import numpy as np


def create_label_list_and_dict(label_file_path: str) -> Tuple[List[str], Dict[str, int]]:
    label_names, label_dict = [], {}
    i = 0
    with open(label_file_path, "r") as file:
        for line in file:
            label_name = line.replace("\n", "")
            label_names.append(label_name)
            label_dict[label_name] = i
            i += 1
    return label_names, label_dict


def create_train_test_indices(data_len: int, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    test_num = int(data_len * test_ratio)
    indices = np.random.permutation(data_len)
    test_indices = indices[:test_num]
    train_indices = indices[test_num:]
    return train_indices, test_indices
