from typing import List
from PIL import Image
import torch
import numpy as np

from .base_model import BaseModel
from ...utils.tensor_util import try_cuda, img_to_tensor
from ...utils.image_utils import *


class SegmentationModel(BaseModel):
    def draw_predicted_result(self, img: Image.Image or torch.Tensor or np.ndarray,
                              img_size_for_model: Tuple[int, int],
                              color_palette: List[int] or None = None, alpha=120) -> Image.Image:
        """
        :param img:
        :param img_size_for_model: Height, Width
        :param color_palette:
        :param alpha:
        :return:
        """
        origin_img_size = get_image_size(img)
        if color_palette is None:
            color_palette = default_color_palette()
        img_tensor = img_to_tensor(resize_image(img, img_size_for_model))
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)
        result = self.predict(img_tensor)[0].argmax(axis=0).astype('uint32')
        result_img = Image.fromarray(result).convert(mode="P")
        if color_palette:
            result_img.putpalette(color_palette)
        result_img = transparent_only_black(result_img, alpha).resize((origin_img_size[1], origin_img_size[0]))
        origin_img = img_to_pil(img)
        return blend_result_and_source(origin_img, np.array(result_img))
