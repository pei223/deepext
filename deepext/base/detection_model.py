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
                              img_size_for_model: Tuple[int, int], label_names: List[str],
                              pred_color=(0, 0, 255)) -> np.ndarray:
        """
        :param img:
        :param img_size_for_model: Height, Width
        :return:
        """
        origin_img_size = get_image_size(img)
        img_tensor = img_to_tensor(resize_image(img, img_size_for_model))
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)

        try:
            result = self.predict(img_tensor)[0]
        except TypeError:
            # print("Bounding box is nothing.")
            result = None
        result = self._scale_bboxes(result, img_size_for_model, origin_img_size)
        img = img_to_cv2(img)
        print(img)
        print(img.shape, img.dtype)
        return self._draw_result_bboxes(img, bboxes_by_class=result, label_names=label_names, pred_color=pred_color)

    def _scale_bboxes(self, bboxes, origin_size: Tuple[int, int], to_size: Tuple[int, int]):
        # TODO
        return bboxes

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
