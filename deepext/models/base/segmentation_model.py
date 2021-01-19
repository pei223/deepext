from typing import Tuple
import torch
import numpy as np
import cv2
from .base_model import BaseModel
from ...utils.tensor_util import try_cuda
from ...utils.image_utils import indexed_image_to_rgb


class SegmentationModel(BaseModel):
    def calc_segmentation_image(self, img: torch.Tensor or np.ndarray, alpha=0.7, require_normalize=False) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        :param img: Numpy array or torch.Tensor image.
        :param require_normalize:
        :param alpha:
        :return: result array(width, height, classes), blended image array(width, height, 3(RGB))
        """
        assert img.ndim == 3, f"Invalid data shape: {img.shape}. Expected 3 dimension"
        if isinstance(img, np.ndarray):
            img_tensor = torch.tensor(img.transpose(2, 0, 1))
            origin_img = img
        elif isinstance(img, torch.Tensor):
            img_tensor = img
            origin_img = img.cpu().numpy().transpose(1, 2, 0)
        else:
            assert False, f"Invalid data type: {type(img)}"

        if require_normalize:
            img_tensor = img_tensor.float() / 255.
        else:
            origin_img = (origin_img * 255).astype('uint8')

        img_tensor = try_cuda(img_tensor)
        img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])

        result_img = self.predict(img_tensor)[0].transpose(1, 2, 0)
        result_index_img = result_img.argmax(axis=-1).astype('uint8')
        result_index_img = indexed_image_to_rgb(result_index_img)

        blended_img = self._blend_img(origin_img, result_index_img, result_alpha=alpha)
        return result_img, blended_img

    def _blend_img(self, origin_img: np.ndarray, result_img: np.ndarray, origin_alpha=1.0,
                   result_alpha=0.7) -> np.ndarray:
        assert origin_img.ndim == 3
        assert result_img.ndim == 3
        return cv2.addWeighted(origin_img, origin_alpha, result_img, result_alpha, 0)
