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
        capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FPS, fps)

        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # frame_size = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)  # 動画書込準備

        self.is_running = True
        while capture.isOpened() and self.is_running:
            ret, frame = capture.read()
            if not ret:
                print("Failed to read frame.")
                break
            result_img = self.calc_result(frame)
            # video_writer.write(result_img)
            cv2.imshow('frame', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Keyboard q pushed.")
                break
        capture.release()
        # video_writer.release()
        cv2.destroyAllWindows()

    @abstractmethod
    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        pass
