from typing import Tuple

import numpy as np
from ..models.base import SegmentationModel
from .realtime_prediction import RealtimePrediction
from ..utils import default_color_palette, pil_to_cv


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)
        self.color_palette = default_color_palette()

    def calc_result(self, frame: np.ndarray):
        result_img = self.model.draw_predicted_result(frame, img_size_for_model=self.img_size_for_model,
                                                      color_palette=self.color_palette)
        return pil_to_cv(result_img)
