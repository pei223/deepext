from typing import Tuple
import cv2
import numpy as np
from ..models.base import SegmentationModel
from .realtime_prediction import RealtimePrediction


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        origin_frame_size = frame.shape[:2]
        frame = cv2.resize(frame, self.img_size_for_model)
        result_img = self.model.calc_segmentation_image(frame, require_normalize=True)[1]
        return cv2.resize(result_img, origin_frame_size)
