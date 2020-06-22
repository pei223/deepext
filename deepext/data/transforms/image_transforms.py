import random
from typing import List, Tuple

import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image


class ImageToOneHot:
    def __init__(self, class_num: int):
        self._class_num = class_num

    def __call__(self, img: torch.Tensor):
        assert img.ndim == 3
        img = img.permute(1, 2, 0).type(torch.LongTensor)
        img = torch.eye(self._class_num)[img]
        return img.view(img.shape[0], img.shape[1], img.shape[3]).permute(2, 0, 1)


class PilToTensor:
    to_tensor = ToTensor()

    def __call__(self, img: Image.Image):
        # NOTE ToTensorでtensor型になるが正規化されてしまうため*255する
        return (self.to_tensor(img) * 255 % 256).long()


class SegmentationLabelSmoothing:
    def __init__(self, class_num: int, epsilon=0.1):
        self._class_num = class_num
        self._epsilon = epsilon
        self._to_onehot = ImageToOneHot(self._class_num)

    def __call__(self, label_img: torch.Tensor):
        assert label_img.ndim == 2 or label_img.ndim == 3
        if label_img.ndim == 2:
            label_img = self._to_onehot(label_img)
        return label_img * (1. - self._epsilon) + self._epsilon


class LabelToOneHot:
    def __init__(self, class_num: int):
        self._class_num = class_num

    def __call__(self, label: torch.Tensor):
        assert label.ndim == 1
        return torch.eye(self._class_num)[label]


class ClassificationLabelSmoothing:
    def __init__(self, class_num: int, epsilon=0.1):
        self._class_num = class_num
        self._epsilon = epsilon
        self._to_onehot = LabelToOneHot(self._class_num)

    def __call__(self, label: torch.Tensor):
        assert label.ndim == 1 or label.ndim == 2
        if label.ndim == 1:
            label = self._to_onehot(label)
        return label * (1. - self._epsilon) + self._epsilon


class LabelAndDataTransforms:
    def __init__(self, transform_sets=List[Tuple[any, any]]):
        """
        :param transform_sets: list of data and label transforms [(data_transform, label_transform), ...]
        """
        self._transform_sets = transform_sets

    def __call__(self, img, label):
        for i in range(len(self._transform_sets)):
            seed = random.randint(0, 2 ** 32)
            data_transform, label_transform = self._transform_sets[i]
            random.seed(seed)
            img = data_transform(img) if data_transform is not None else img
            random.seed(seed)
            label = label_transform(label) if label_transform is not None else label
        return img, label
