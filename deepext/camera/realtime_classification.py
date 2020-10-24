from typing import Tuple, List
import numpy as np
from ..models.base import ClassificationModel, AttentionClassificationModel
from .realtime_prediction import RealtimePrediction
from ..utils.image_utils import *


class RealtimeClassification(RealtimePrediction):
    def __init__(self, model: ClassificationModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, ClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        label = self.model.predict_label(frame, self.img_size_for_model)
        result_text = f"Inference result:    {self._label_names[label]}"
        offsets = (0, 40)
        background_color = (255, 255, 255)
        text_color = (0, 0, 255)
        return draw_text_with_background(frame, background_color=background_color, text_color=text_color,
                                         text=result_text, offsets=offsets, font_scale=0.5)


class RealtimeAttentionClassification(RealtimePrediction):
    def __init__(self, model: AttentionClassificationModel, img_size_for_model: Tuple[int, int],
                 label_names: List[str]):
        assert isinstance(model, AttentionClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        origin_frame_size = frame.shape[:2]
        frame = cv2.resize(frame, self.img_size_for_model)
        label, heatmap = self.model.predict_label_and_heatmap(frame, require_normalize=True)
        result_img = cv2.resize(heatmap, origin_frame_size)
        result_text = f"Inference result:    {self._label_names[label]}"
        offsets = (0, 80)
        background_color = (255, 255, 255)
        text_color = (0, 0, 255)
        # TODO calc_result終わった後に文章を載せないとぼやける
        return draw_text_with_background(result_img, background_color=background_color, text_color=text_color,
                                         text=result_text, offsets=offsets, font_scale=0.5)
