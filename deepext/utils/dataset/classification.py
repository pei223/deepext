from collections import OrderedDict
from warnings import warn
from torch.utils.data import Dataset
import csv
from PIL import Image
from pathlib import Path


class CSVAnnotationDataset(Dataset):
    def __init__(self, image_dir: str, annotation_filepath: str, image_transform, label_transform=None):
        self._image_dir = image_dir
        self._image_transform = image_transform
        self._label_transform = label_transform
        self._filepath_label_dict = OrderedDict()
        with open(annotation_filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row[1].isdigit():
                    warn(f"invalid label: {row}")
                    continue
                self._filepath_label_dict[row[0]] = int(row[1])
        self._filepath_ls = list(self._filepath_label_dict.keys())

    def __len__(self):
        return len(self._filepath_label_dict)

    def __getitem__(self, idx):
        filepath, label = list(self._filepath_label_dict.items())[idx]
        filepath = Path(self._image_dir).joinpath(filepath)
        img = Image.open(str(filepath))
        img = img.convert("RGB")
        if self._image_transform:
            img = self._image_transform(img)
        if self._label_transform:
            label = self._label_transform(img)
        return img, label
