from typing import List, Union, Tuple
import torch
import numpy as np
import cv2

from .base_model import BaseModel
from ...utils.tensor_util import try_cuda
from ...utils.image_utils import draw_bounding_boxes_with_name_tag


class DetectionModel(BaseModel):
    def calc_detection_image(self, img: torch.Tensor or np.ndarray, label_names: List[str],
                             origin_img_size: Tuple[int, int] = None, require_normalize=False,
                             pred_bbox_color=(0, 0, 255)) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        :param img: Numpy array or torch.Tensor image.
        :param label_names:
        :param origin_img_size:
        :param require_normalize:
        :param pred_bbox_color:
        :return: Object detection result(bounding box num, 5(x_min, y_min, x_max, y_max, label)), blended image.
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
        model_img_size = (img_tensor.shape[-2], img_tensor.shape[-1])

        try:
            result = self.predict(img_tensor)[0]
        except TypeError:
            result = None
        if origin_img_size:
            origin_img = cv2.resize(origin_img, origin_img_size)
            result = self._scale_bboxes(result, model_img_size, origin_img_size)
        return result, self._draw_result_bboxes(origin_img, bboxes=result, label_names=label_names,
                                                pred_color=pred_bbox_color)

    def _scale_bboxes(self, bboxes: Union[np.ndarray, List[List[float or int]]], origin_size: Tuple[int, int],
                      to_size: Tuple[int, int]) -> Union[np.ndarray, List[List[float or int]]]:
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

    def _draw_result_bboxes(self, image: np.ndarray, bboxes: Union[np.ndarray, List[List[float or int]]],
                            label_names: List[str], pred_color=(0, 0, 255)):
        if bboxes is None:
            return image
        for bbox in bboxes:
            if bbox is None or len(bbox) == 0:
                continue
            score = int(bbox[5] * 100.)
            label_name = label_names[int(bbox[4])]
            bbox_text = f"{label_name} {score}%"
            image = draw_bounding_boxes_with_name_tag(image, bboxes, color=pred_color,
                                                      text=bbox_text)
        return image
