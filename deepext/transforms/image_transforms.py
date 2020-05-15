import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image


class ImageToOneHot:
    def __init__(self, class_num: int):
        self._class_num = class_num

    def __call__(self, img: torch.Tensor):
        img = img.permute(1, 2, 0).type(torch.LongTensor)
        img = torch.eye(self._class_num)[img]
        return img.view(img.shape[0], img.shape[1], img.shape[3]).permute(2, 0, 1)


class PilToTensor:
    to_tensor = ToTensor()

    def __call__(self, img: Image.Image):
        # NOTE ToTensorでtensor型になるが正規化されてしまうため*255する
        return (self.to_tensor(img) * 255 % 256).long()
