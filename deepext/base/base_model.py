from abc import ABCMeta, abstractmethod
import torch
import numpy as np


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        1バッチの学習を進めてlossを返却する.
        :param train_x: B * Channel * H * W
        :param teacher: B * Class * H * W
        :return:
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def save_weight(self, save_path: str):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def load_weight(self, weight_path: str):
        pass
