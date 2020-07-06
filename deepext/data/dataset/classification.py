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
                filename = row[0]
                label = row[1]
                if not label.isdigit():
                    warn(f"invalid label: {row}")
                    continue
                self._filepath_label_dict[filename] = int(label)

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
