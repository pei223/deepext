from typing import Tuple
from PIL import Image
import torch
import numpy as np

from .base_model import BaseModel
from ...utils.tensor_util import try_cuda, img_to_tensor
from ...utils.image_utils import *


class SegmentationModel(BaseModel):
    def draw_predicted_result(self, img: Image.Image or torch.Tensor or np.ndarray,
                              img_size_for_model: Tuple[int, int], alpha=0.7) -> np.ndarray:
        """
        :param img:
        :param img_size_for_model: Height, Width
        :param alpha:
        :return:
        """
        origin_img_size = get_image_size(img)
        img_tensor = img_to_tensor(resize_image(img, img_size_for_model))
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)
        result_img = self.predict(img_tensor)[0].argmax(axis=0).astype('uint32')
        result_img = indexed_image_to_rgb(result_img)

        # 0~1のTensorを0~255に変換
        origin_img = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        origin_img = (origin_img * 255).astype("uint8")

        blended_img = self._blend_img(origin_img, result_img, result_alpha=alpha)
        return cv2.resize(blended_img, origin_img_size)

    def _blend_img(self, origin_img: np.ndarray, result_img: np.ndarray, origin_alpha=1.0,
                   result_alpha=0.6) -> np.ndarray:
        assert origin_img.ndim == 3
        assert result_img.ndim == 3
        return cv2.addWeighted(origin_img, origin_alpha, result_img, result_alpha, 0)
