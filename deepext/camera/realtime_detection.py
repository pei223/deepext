from typing import Tuple, List

import cv2
from ..base import DetectionModel
import numpy as np


class RealtimeDetection:
    def __init__(self, model: DetectionModel, img_size: Tuple[int, int], label_names: List[str]):
        self.model = model
        self.is_running = False
        self.img_size = img_size
        self.label_names = label_names

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
            result_img = self.calc_detection_result(frame)
            # out.write(result_img)
            cv2.imshow('frame', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Keyboard q pushed.")
                break
        video.release()
        # out.release()
        cv2.destroyAllWindows()

    def calc_detection_result(self, frame: np.ndarray):
        origin_img_size = frame.shape[:2]
        frame = cv2.resize(frame, dsize=self.img_size)
        return self.model.draw_predicted_result(frame, size=origin_img_size, label_names=self.label_names)
