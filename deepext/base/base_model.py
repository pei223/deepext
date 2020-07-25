from abc import ABCMeta, abstractmethod
import torch
import numpy as np
import torch.nn as nn


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

    # TODO state_dict以外の保存方法など考える
    # @abstractmethod
    # def get_model(self) -> nn.Module:
    #     pass
    #
    # def save_model(self, filepath: str):
    #     torch.save(self.get_model(), filepath)
    #
    # def load_model(self, filepath: str) -> nn.Module:
    #     return torch.load(filepath)
