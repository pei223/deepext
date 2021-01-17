from collections import OrderedDict
from typing import Dict, Tuple, List, Generator
from warnings import warn
from torch.utils.data import Dataset
import csv
from PIL import Image
from pathlib import Path
import numpy as np

from .dataset_factory import MultipleDatasetFactory


class CSVAnnotationMultiDatasetFactory(MultipleDatasetFactory):
    def __init__(self,
                 images_dir: str, annotation_csv_filepath: str,
                 train_image_transform, test_image_transform, train_label_transform=None,
                 test_label_transform=None, label_dict: Dict[str, int] = None):
        self._images_dir = images_dir
        self._annotation_csv_filepath = annotation_csv_filepath
        self._train_img_transform, self._test_img_transform = train_image_transform, test_image_transform
        self._train_label_transform, self._test_label_transform = train_label_transform, test_label_transform
        self._label_dict = label_dict

    def create_train_test(self, train_indices: np.ndarray, test_indices: np.ndarray) -> Tuple[Dataset, Dataset]:
        filename_label_dict = CSVAnnotationDataset.create_filename_label_dict(self._annotation_csv_filepath,
                                                                              self._label_dict)
        filename_ls = list(filename_label_dict.keys())

        train_filename_label_dict, test_filename_label_dict = OrderedDict(), OrderedDict()

        for idx in train_indices:
            filepath = filename_ls[idx]
            label = filename_label_dict[filepath]
            train_filename_label_dict[filepath] = label
        for idx in test_indices:
            filepath = filename_ls[idx]
            label = filename_label_dict[filepath]
            test_filename_label_dict[filepath] = label
        train_dataset = CSVAnnotationDataset(image_dir=self._images_dir,
                                             image_transform=self._train_img_transform,
                                             label_transform=self._train_label_transform,
                                             filename_label_dict=train_filename_label_dict)
        test_dataset = CSVAnnotationDataset(image_dir=self._images_dir,
                                            image_transform=self._test_img_transform,
                                            label_transform=self._test_label_transform,
                                            filename_label_dict=test_filename_label_dict)
        return train_dataset, test_dataset

    def create_kfold_generator(self, k_indices_ls: List[np.ndarray]) -> Generator[Tuple[Dataset, Dataset], None, None]:
        assert len(k_indices_ls) >= 2
        for i in range(len(k_indices_ls) - 1):
            train_indices, test_indices = self.split_kfold_train_test_indices(k_indices_ls, i)
            train_dataset, test_dataset = self.create_train_test(train_indices, test_indices)
            yield train_dataset, test_dataset


class CSVAnnotationDataset(Dataset):
    @staticmethod
    def create(image_dir: str, annotation_csv_filepath: str, image_transform,
               label_transform=None, label_dict: Dict[str, int] = None) -> 'CSVAnnotationDataset':
        """
        Create dataset from CSV file.
        If CSV column 2 value is string, required label_dict arg.
        :param label_dict:
        :param image_dir:
        :param annotation_csv_filepath: 
        :param image_transform:
        :param label_transform: 
        :return: 
        """
        filename_label_dict = CSVAnnotationDataset.create_filename_label_dict(annotation_csv_filepath, label_dict)
        return CSVAnnotationDataset(image_dir, filename_label_dict=filename_label_dict, image_transform=image_transform,
                                    label_transform=label_transform)

    @staticmethod
    def create_filename_label_dict(annotation_csv_filepath: str, label_dict: Dict[str, int] = None) -> OrderedDict:
        filename_label_dict = OrderedDict()
        with open(annotation_csv_filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                filename = Path(row[0]).name
                label = row[1]
                if not label.isdigit():
                    if label_dict is None:
                        raise RuntimeError("Required dict transform label name to class number.")
                    label_num = label_dict.get(label)
                    if label_num is None:
                        warn(f"Invalid label name: {label},  {row}")
                        continue
                    filename_label_dict[filename] = label_num
                    continue
                filename_label_dict[filename] = int(label)
        return filename_label_dict

    def __init__(self, image_dir: str, filename_label_dict: OrderedDict, image_transform,
                 label_transform=None):
        self._image_dir = image_dir
        self._image_transform = image_transform
        self._label_transform = label_transform
        self._filepath_label_dict = filename_label_dict

    def __len__(self):
        return len(self._filepath_label_dict)

    def __getitem__(self, idx):
        filename, label = list(self._filepath_label_dict.items())[idx]
        filepath = Path(self._image_dir).joinpath(filename)
        img = Image.open(str(filepath))
        img = img.convert("RGB")
        if self._image_transform:
            img = self._image_transform(img)
        if self._label_transform:
            label = self._label_transform(img)
        return img, label
