from typing import Tuple

import numpy as np
from ..models.base import SegmentationModel
from .realtime_prediction import RealtimePrediction


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        return self.model.draw_predicted_result(frame, img_size_for_model=self.img_size_for_model)
