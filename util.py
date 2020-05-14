from typing import Tuple, List


class DataSetSetting:
    def __init__(self, dataset_type: str, size: Tuple[int, int], n_classes: int, label_names: List[str] = None):
        self.dataset_type = dataset_type
        self.size = size
        self.n_classes = n_classes
        self.label_names = label_names

    @staticmethod
    def from_dataset_type(dataset_setting_ls: List, dataset_type: str):
        dataset_setting_type_names = []
        for dataset_setting in dataset_setting_ls:
            if dataset_setting.dataset_type == dataset_type:
                return dataset_setting
            dataset_setting_type_names.append(dataset_setting.dataset_type)
        assert f"Invalid dataset type. Valid dataset is {dataset_setting_type_names}"
