from typing import List
from PIL import Image
import torch
import numpy as np
import cv2

from .base_model import BaseModel
from ..utils.tensor_util import try_cuda, img_to_tensor
from ..utils.image_utils import *


class DetectionModel(BaseModel):
    def draw_predicted_result(self, img: Image.Image or torch.Tensor or np.ndarray,
                              size: Tuple[int, int], label_names: List[str], pred_color=(0, 0, 255)) -> np.ndarray:
        """
        :param img:
        :param size: Height, Width
        :return:
        """
        img_tensor = img_to_tensor(img)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)

        result = self.predict(img_tensor)[0].argmax(axis=0)
        img = img_to_cv2(img)
        img = self._draw_result_bboxes(img, bboxes_by_class=result, label_names=label_names, pred_color=pred_color)
        return cv2.resize(img, dsize=size)

    def _draw_result_bboxes(self, image: np.ndarray, bboxes_by_class: List[List[float]], label_names: List[str],
                            pred_color=(0, 0, 255)):
        if bboxes_by_class is None:
            return image
        for i, bboxes in enumerate(bboxes_by_class):
            if bboxes is None or len(bboxes) == 0:
                continue
            image = draw_bounding_boxes_with_name_tag(image, bboxes, color=pred_color,
                                                      text=label_names[i])
        return image
