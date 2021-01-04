from collections import OrderedDict
from typing import Dict
from warnings import warn
from torch.utils.data import Dataset
import csv
from PIL import Image
from pathlib import Path


class CSVAnnotationDataset(Dataset):
    @staticmethod
    def create_from_label_val(image_dir: str, annotation_csv_filepath: str, image_transform,
                              label_transform=None) -> 'CSVAnnotationDataset':
        """
        Create dataset from CSV file.
        CSV column 2 must be integer(label value).
        :param image_dir: 
        :param annotation_csv_filepath: 
        :param image_transform:
        :param label_transform: 
        :return: 
        """
        filename_label_dict = OrderedDict()
        with open(annotation_csv_filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                filename = Path(row[0]).name
                label = row[1]
                if not label.isdigit():
                    warn(f"invalid label: {row}")
                    continue
                filename_label_dict[filename] = int(label)
        return CSVAnnotationDataset(image_dir, filename_label_dict=filename_label_dict, image_transform=image_transform,
                                    label_transform=label_transform)

    @staticmethod
    def create_from_label_name(image_dir: str, annotation_csv_filepath: str, label_dict: Dict[str, int], image_transform,
                               label_transform=None) -> 'CSVAnnotationDataset':
        """
        Create dataset from CSV file.
        CSV column 2 must be str(label name).
        :param label_dict: Key is label name(string). Value is label value(integer).
        :param image_dir: 
        :param annotation_csv_filepath: 
        :param image_transform: 
        :param label_transform: 
        :return: 
        """
        filename_label_dict = OrderedDict()
        with open(annotation_csv_filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                filename = Path(row[0]).name
                label = label_dict.get(row[1])
                if label is None:
                    warn(f"invalid label: {row}")
                    continue
                filename_label_dict[filename] = label
        return CSVAnnotationDataset(image_dir, filename_label_dict=filename_label_dict, image_transform=image_transform,
                                    label_transform=label_transform)

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
