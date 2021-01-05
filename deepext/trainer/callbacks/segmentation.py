import numpy as np
from torch.utils.data import Dataset
from ...models.base import SegmentationModel
from ...utils import image_utils


class GenerateSegmentationImageCallback:
    def __init__(self, model: SegmentationModel, output_dir: str, per_epoch: int, dataset: Dataset, alpha=0.6,
                 apply_all_images=False):
        self._model: SegmentationModel = model
        self._output_dir = output_dir
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._alpha = alpha
        self._apply_all_images = apply_all_images

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        if self._apply_all_images:
            i = 1
            for img_tensor, label in self._dataset:
                _, result_img = self._model.calc_segmentation_image(img_tensor, alpha=self._alpha)
                image_utils.cv_to_pil(result_img).save(f"{self._output_dir}/data{i}_image{epoch + 1}.png")
                i += 1
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, _ = self._dataset[random_image_index]
        _, result_img = self._model.calc_segmentation_image(img_tensor, alpha=self._alpha)
        image_utils.cv_to_pil(result_img).save(f"{self._output_dir}/result_image{epoch + 1}.png")
