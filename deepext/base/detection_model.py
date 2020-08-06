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
        return self._draw_result_bboxes(img, bboxes_by_class=result, label_names=label_names, pred_color=pred_color)

    def _scale_bboxes(self, bboxes_by_class: List[List[List[float]]], origin_size: Tuple[int, int],
                      to_size: Tuple[int, int]):
        height_rate = to_size[0] / origin_size[0]
        width_rate = to_size[1] / origin_size[1]
        if bboxes_by_class is None:
            return bboxes_by_class
        for cls in range(len(bboxes_by_class)):
            print(bboxes_by_class[cls])
            if bboxes_by_class[cls] is None or len(bboxes_by_class[cls]) == 0:
                continue
            for i in range(len(bboxes_by_class[cls])):
                bboxes_by_class[cls][i][0], bboxes_by_class[cls][i][2] = bboxes_by_class[cls][i][0] * width_rate, \
                                                                         bboxes_by_class[cls][i][
                                                                             2] * width_rate
                bboxes_by_class[cls][i][1], bboxes_by_class[cls][i][3] = bboxes_by_class[cls][i][1] * height_rate, \
                                                                         bboxes_by_class[cls][i][
                                                                             3] * height_rate
        return bboxes_by_class

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
