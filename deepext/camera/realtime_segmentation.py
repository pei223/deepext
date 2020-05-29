from typing import Tuple

import cv2
from ..base import SegmentationModel
import numpy as np
from ..utils import cv2image_to_tensor, default_color_palette, transparent_only_black, blend_result_and_source, \
    pil_to_cv, cv_to_pil


class RealtimeSegmentation:
    def __init__(self, model: SegmentationModel, img_size: Tuple[int, int]):
        self.model = model
        self.is_running = False
        self.color_palette = default_color_palette()
        self.img_size = img_size

    def stop(self):
        self.is_running = False

    def realtime_predict(self, device_id=0, fps=10):
        video = cv2.VideoCapture(device_id)
        video.set(cv2.CAP_PROP_FPS, fps)

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # out = cv2.VideoWriter('./output.mp4', apiPreference=cv2.CAP_INTEL_MFX, fourcc=fourcc, fps=fps,
        #                       frameSize=frame_size)  # 動画書込準備

        self.is_running = True
        while video.isOpened() and self.is_running:
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame.")
                break
            result_img = self.calc_segmentation_result(frame)
            # out.write(result_img)
            cv2.imshow('frame', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Keyboard q pushed.")
                break
        video.release()
        # out.release()
        cv2.destroyAllWindows()

    def calc_segmentation_result(self, frame: np.ndarray):
        origin_img_size = frame.shape[:2]
        frame = cv2.resize(frame, dsize=self.img_size)
        result_img = self.model.draw_predicted_result(frame, size=origin_img_size, color_palette=self.color_palette)
        return pil_to_cv(result_img)
