from typing import List
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class IndexImageDataset(Dataset):
    def __init__(self, image_dir_path: str, index_image_dir_path: str, transforms, valid_suffixes: List[str] = None):
        """
        :param image_dir_path: Directory path of images.
        :param index_image_dir_path: Directory path of index images.
        :param transforms: LabelAndDataTransforms or Albumentations.Compose
        :param valid_suffixes:
        """
        self.transforms = transforms
        image_dir = Path(image_dir_path)

        if valid_suffixes is None:
            valid_suffixes = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        self.image_path_ls = []
        for suffix in valid_suffixes:
            self.image_path_ls += list(image_dir.glob(suffix))
        self.index_image_dir = Path(index_image_dir_path)
        self.image_path_ls.sort()

    def __call__(self, idx: int):
        image_path = self.image_path_ls[idx]
        index_image_path = self.index_image_dir.joinpath(f"{image_path.stem}.png")
        image = Image.open(str(image_path)).convert("RGB")
        index_image = Image.open(str(index_image_path))
        return self.transforms(image, index_image)
