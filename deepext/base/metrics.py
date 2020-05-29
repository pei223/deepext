from abc import abstractmethod

import numpy as np
import torch


class Metrics:
    @abstractmethod
    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        1バッチの指標を算出しメンバ変数に保持する.
        :param pred: prediction result. (Batch size, ...)
        :param teacher: teacher data. (Batch size, ...)
        """
        pass

    @abstractmethod
    def calc_summary(self) -> any:
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

    def save_visualized_metrics(self, result, filepath: str, epoch: int):
        """
        指標を可視化して保存する
        :param result: 指標値
        :param filepath: 保存先ファイルパス
        :param epoch:
        """
        pass
