from typing import Tuple, List

import numpy as np
from .realtime_prediction import RealtimePrediction
from ..models.base import DetectionModel


class RealtimeDetection(RealtimePrediction):
    def __init__(self, model: DetectionModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, DetectionModel)
        super().__init__(model, img_size_for_model)
        self.label_names = label_names

    def calc_result(self, frame: np.ndarray):
        return self.model.draw_predicted_result(frame, img_size_for_model=self.img_size_for_model,
                                                label_names=self.label_names)
