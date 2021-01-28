from typing import Tuple
from abc import abstractmethod
import torch
import numpy as np
import cv2
from .base_model import BaseModel
from ...utils.tensor_util import try_cuda
from ...utils.image_utils import combine_heatmap


class ClassificationModel(BaseModel):
    def predict_label(self, img: torch.Tensor or np.ndarray, require_normalize=False) -> Tuple[int, np.ndarray]:
        """
        :param img: (batch size, channels, height, width)
        :param require_normalize:
        :return: Class num, probabilities array
        """
        # NOTE Override if needed.
        result = self._predict(img, require_normalize)
        result = self._numpy_softmax(result)
        label = np.argmax(result)
        return label, result

    def predict_multi_class(self, img: torch.Tensor or np.ndarray, threshold=0.5, require_normalize=False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        :param img: (batch size, channels, height, width)
        :param threshold:
        :param require_normalize:
        :return: Class num array, probabilities array
        """
        result = self._predict(img, require_normalize)
        result = self._numpy_sigmoid(result)
        class_result = np.where(result > threshold)[0]
        return class_result, result

    def _predict(self, img: torch.Tensor or np.ndarray, require_normalize):
        """
        :param img: (batch size, channels, height, width)
        :return: Class num
        """
        # NOTE Override if needed.
        assert img.ndim == 3, f"Invalid data shape: {img.shape}. Expected 3 dimension"
        if isinstance(img, np.ndarray):
            img_tensor = torch.tensor(img.transpose(2, 0, 1))
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            assert False, f"Invalid data type: {type(img)}"
        if require_normalize:
            img_tensor = img_tensor.float() / 255.
        img_tensor = try_cuda(img_tensor)
        img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        return self.predict(img_tensor)[0]

    def _numpy_softmax(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        u = np.sum(np.exp(x))
        return np.exp(x) / u

    def _numpy_sigmoid(self, x) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class AttentionClassificationModel(ClassificationModel):
    def predict_label_and_heatmap(self, img: torch.Tensor or np.ndarray, require_normalize=False, origin_alpha=0.6) -> \
            Tuple[int, np.ndarray]:
        """
        :param origin_alpha:
        :param require_normalize:
        :param img:
        :return: Class num, heatmap
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

        label_array, heatmap_img = self.predict_label_and_heatmap_impl(img_tensor)

        heatmap_img = cv2.resize((heatmap_img[0] * 255).astype("uint8"), origin_img.shape[:2])

        blended_img = combine_heatmap(origin_img, heatmap_img, origin_alpha=origin_alpha)
        return label_array[0].argmax(), blended_img

    @abstractmethod
    def predict_label_and_heatmap_impl(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: inputs: (batch size, channels, height, width)
        :return: Class num, heatmap
        """
        pass
