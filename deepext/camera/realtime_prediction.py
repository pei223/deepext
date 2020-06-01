from abc import abstractmethod
from typing import Tuple

import cv2
from ..base import BaseModel
import numpy as np


class RealtimePrediction:
    def __init__(self, model: BaseModel, img_size_for_model: Tuple[int, int]):
        self.model = model
        self.is_running = False
        self.img_size_for_model = img_size_for_model

    def stop(self):
        self.is_running = False

    def realtime_predict(self, device_id=0, fps=10):
        video = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
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
            result_img = self.calc_result(frame)
            # out.write(result_img)
            cv2.imshow('frame', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Keyboard q pushed.")
                break
        video.release()
        # out.release()
        cv2.destroyAllWindows()

    @abstractmethod
    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        pass
