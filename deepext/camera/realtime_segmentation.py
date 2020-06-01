from typing import Tuple

import cv2
import numpy as np
from ..base import SegmentationModel
from .realtime_prediction import RealtimePrediction
from ..utils import cv2image_to_tensor, default_color_palette, transparent_only_black, blend_result_and_source, \
    pil_to_cv, cv_to_pil


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)
        self.color_palette = default_color_palette()

    def calc_result(self, frame: np.ndarray):
        result_img = self.model.draw_predicted_result(frame, img_size_for_model=self.img_size_for_model,
                                                      color_palette=self.color_palette)
        return pil_to_cv(result_img)
