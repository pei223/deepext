from typing import List

import numpy
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from ...base import SegmentationModel
from ...utils import *


class GenerateSegmentationImageCallback:
    def __init__(self, model: SegmentationModel, output_dir: str, per_epoch: int, dataset: Dataset, alpha=150,
                 color_palette_ls: List[int] = None):
        self._model: SegmentationModel = model
        self._output_dir = output_dir
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._alpha = alpha
        self._to_pil = ToPILImage()
        self._color_palette = color_palette_ls or default_color_palette()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, _ = self._dataset[random_image_index]

        result_img = self._model.draw_predicted_result(img_tensor, img_size_for_model=img_tensor.shape[1:],
                                                       color_palette=self._color_palette)
        result_img.save(
            f"{self._output_dir}/result_image{epoch + 1}.png")
