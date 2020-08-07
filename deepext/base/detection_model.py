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
        return self._draw_result_bboxes(img, bboxes=result, label_names=label_names, pred_color=pred_color)

    def _scale_bboxes(self, bboxes: List[List[float]], origin_size: Tuple[int, int],
                      to_size: Tuple[int, int]):
        height_rate = to_size[0] / origin_size[0]
        width_rate = to_size[1] / origin_size[1]
        if bboxes is None:
            return bboxes
        for i in range(len(bboxes)):
            if bboxes[i] is None or len(bboxes[i]) == 0:
                continue
            bboxes[i][0], bboxes[i][2] = bboxes[i][0] * width_rate, bboxes[i][2] * width_rate
            bboxes[i][1], bboxes[i][3] = bboxes[i][1] * height_rate, bboxes[i][3] * height_rate
        return bboxes

    def _draw_result_bboxes(self, image: np.ndarray, bboxes: List[List[float or int]], label_names: List[str],
                            pred_color=(0, 0, 255)):
        if bboxes is None:
            return image
        for bbox in bboxes:
            if bbox is None or len(bbox) == 0:
                continue
            image = draw_bounding_boxes_with_name_tag(image, bboxes, color=pred_color,
                                                      text=label_names[int(bbox[-1])])
        return image
