from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from . import ModelCallback
from ...models.base import DetectionModel
from ...utils import image_utils, draw_bounding_boxes_with_name_tag


class GenerateDetectionImageCallback(ModelCallback):
    def __init__(self, model: DetectionModel, img_size: Tuple[int, int], dataset: Dataset, out_dir: str,
                 label_names: List[str], per_epoch: int = 10, pred_color=(0, 0, 255), teacher_color=(0, 255, 0),
                 apply_all_images=False):
        """
        :param model:
        :param img_size: (H, W)
        :param dataset:
        :param out_dir:
        :param per_epoch:
        :param pred_color:
        :param teacher_color:
        """
        self._model = model
        self._dataset = dataset
        self._pred_color = pred_color
        self._teacher_color = teacher_color
        self._per_epoch = per_epoch
        self._out_dir = out_dir
        self._img_size = img_size
        self._label_names = label_names
        self._apply_all_images = apply_all_images
        if not Path(self._out_dir).exists():
            Path(self._out_dir).mkdir()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        if self._apply_all_images:
            for i, (img_tensor, teacher_bboxes) in enumerate(self._dataset):
                _, result_img = self._model.calc_detection_image(img_tensor, label_names=self._label_names)
                result_img = self._draw_teacher_bboxes(result_img, teacher_bboxes=teacher_bboxes)
                image_utils.cv_to_pil(result_img).save(f"{self._out_dir}/data{i}_image{epoch + 1}.png")
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        image, teacher_bboxes = self._dataset[random_image_index]
        result_img = self._model.calc_detection_image(image, label_names=self._label_names)[1]
        result_img = self._draw_teacher_bboxes(result_img, teacher_bboxes=teacher_bboxes)
        image_utils.cv_to_pil(result_img).save(f"{self._out_dir}/result_{epoch + 1}.png")

    def _draw_teacher_bboxes(self, image: np.ndarray, teacher_bboxes: List[Tuple[float, float, float, float, int]]):
        """
        :param image:
        :param teacher_bboxes: List of [x_min, y_min, x_max, y_max, label]
        :return:
        """
        if teacher_bboxes is None or len(teacher_bboxes) == 0:
            return image
        for bbox in teacher_bboxes:
            image = draw_bounding_boxes_with_name_tag(image, [bbox], color=self._teacher_color, text=None)
        return image
