from typing import Tuple
from abc import abstractmethod
import numpy as np
from .base_model import BaseModel
from ...utils.tensor_util import try_cuda, img_to_tensor
from ...utils.image_utils import *


class ClassificationModel(BaseModel):
    def predict_label(self, img: Image.Image or torch.Tensor or np.ndarray,
                      img_size_for_model: Tuple[int, int]) -> int:
        """
        :param img: (batch size, channels, height, width)
        :return: Class num
        """
        # NOTE Override if needed.
        img_tensor = img_to_tensor(resize_image(img, img_size_for_model))
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)

        return self.predict(img_tensor)[0].argmax()


class AttentionClassificationModel(ClassificationModel):
    def predict_label_and_heatmap(self, img: Image.Image or torch.Tensor or np.ndarray,
                                  img_size_for_model: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        """
        :param img:
        :param img_size_for_model:
        :return: Class num, heatmap
        """
        origin_img_size = get_image_size(img)
        img_tensor = img_to_tensor(resize_image(img, img_size_for_model))
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = try_cuda(img_tensor)
        label_array, heatmap_img = self.predict_label_and_heatmap_impl(img_tensor)

        origin_img = img_to_cv2(img)
        heatmap_img = cv2.resize((heatmap_img[0] * 255).astype("uint8"), origin_img_size[::-1])

        blended_img = combine_heatmap(origin_img, heatmap_img, origin_alpha=0.5)
        return label_array[0].argmax(), blended_img

    @abstractmethod
    def predict_label_and_heatmap_impl(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: inputs: (batch size, channels, height, width)
        :return: Class num, heatmap
        """
        pass
