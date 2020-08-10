from typing import List
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ...models.base import SegmentationModel
from ...utils import try_cuda, image_utils


class GenerateSegmentationImageCallback:
    def __init__(self, model: SegmentationModel, output_dir: str, per_epoch: int, dataset: Dataset, alpha=0.6):
        self._model: SegmentationModel = model
        self._output_dir = output_dir
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._alpha = alpha

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, _ = self._dataset[random_image_index]
        img_tensor = try_cuda(img_tensor)

        result_img = self._model.draw_predicted_result(img_tensor, img_size_for_model=img_tensor.shape[1:],
                                                       alpha=self._alpha)
        Image.fromarray(result_img).save(f"{self._output_dir}/result_image{epoch + 1}.png")
