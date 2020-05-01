import torch
import numpy as np


class ImageToOneHot:
    def __init__(self, class_num: int):
        self._class_num = class_num

    def __call__(self, img: torch.Tensor):
        img = img.permute(1, 2, 0).type(torch.LongTensor)
        img = torch.eye(self._class_num)[img]
        return img.view(img.shape[0], img.shape[1], img.shape[3]).permute(2, 0, 1)


class ReturnFloatTensorToInt:
    def __call__(self, normalized_img: torch.Tensor):
        test = (normalized_img * 255 % 255).long()
        return test
