from abc import abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch


class BaseMetric:
    @abstractmethod
    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        1バッチの指標を算出しメンバ変数に保持する.
        :param pred: prediction result. (Batch size, ...)
        :param teacher: teacher data. (Batch size, ...)
        """
        pass

    @abstractmethod
    def calc_summary(self) -> Tuple[float, Dict[str, float]]:
        """
        データ全体の指標を計算して返却する.
        :return: 指標値.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        メンバ変数に保持している指標をクリアする.
        """
        pass

    @abstractmethod
    def clone_empty(self) -> 'BaseMetric':
        pass

    @abstractmethod
    def clone(self) -> 'BaseMetric':
        pass

    @abstractmethod
    def __add__(self, other: 'BaseMetric') -> 'BaseMetric':
        pass

    @abstractmethod
    def __truediv__(self, num: int) -> 'BaseMetric':
        pass
