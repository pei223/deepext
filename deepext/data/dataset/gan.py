from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class SimpleGANDataSet(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.filepaths = list(Path(root_dir).glob("*.png")) + list(Path(root_dir).glob("*.jpg")) + list(
            Path(root_dir).glob("*.jpeg"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(str(self.filepaths[idx]))
        if self.transform:
            img = self.transform(img)
        return img