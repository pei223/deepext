from typing import Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ...models.base import DetectionModel
from ...utils import *


class VisualizeRandomObjectDetectionResult:
    def __init__(self, model: DetectionModel, img_size: Tuple[int, int], dataset: Dataset, out_dir: str,
                 label_names: List[str], per_epoch: int = 10, pred_color=(0, 0, 255), teacher_color=(0, 255, 0)):
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

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        image, teacher_bboxes = self._dataset[random_image_index]
        assert isinstance(image, torch.Tensor), "Expected image type is Tensor."
        result_img = self._model.calc_detection_image(image, label_names=self._label_names)[1]
        image = self._draw_teacher_bboxes(result_img, teacher_bboxes=teacher_bboxes)
        cv2.imwrite(f"{self._out_dir}/result_{epoch + 1}.png", image)

    def _draw_teacher_bboxes(self, image: np.ndarray, teacher_bboxes: List[Tuple[float, float, float, float, int]]):
        """
        :param image:
        :param teacher_bboxes: List of [x_min, y_min, x_max, y_max, label]
        :return:
        """
        if teacher_bboxes is None or len(teacher_bboxes) == 0:
            return image
        for bbox in teacher_bboxes:
            image = draw_bounding_boxes_with_name_tag(image, [bbox], color=self._teacher_color,
                                                      text=self._label_names[int(bbox[-1])])
        return image
