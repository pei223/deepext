from torch.utils.data import DataLoader
from typing import List, Callable
from statistics import mean
import numpy as np
import time

from deepext.base import BaseModel
from deepext.utils.tensor_util import try_cuda
from ..base import Metrics
from .lrcurve_visualizer import LRCurveVisualizer
import torch


class Trainer:
    def __init__(self, model: BaseModel):
        self._model: BaseModel = model

    def fit(self, data_loader: DataLoader, epochs: int, test_dataloader: DataLoader = None,
            callbacks: List[Callable[[int, ], None]] = None, metric_ls: List[Metrics] = None,
            lr_scheduler_func: Callable[[int, ], float] = None, calc_metrics_per_epoch: int = 5,
            lr_graph_filepath: str = None, metric_for_graph: Metrics = None):
        """
        :param data_loader: DataLoader for training
        :param epochs:
        :param test_dataloader: DataLoader for test
        :param callbacks:
        :param metric_ls: 指標リスト
        :param lr_scheduler_func: 学習率スケジューリング関数
        :param calc_metrics_per_epoch: 何エポックごとに指標を計算するか
        :param lr_graph_filepath: 学習曲線保存先ファイルパス. Noneなら描画なし
        :param metric_for_graph: 学習曲線に使用する指標クラス. スカラー値を返す指標である必要がある
        """
        callbacks, metric_ls = callbacks or [], metric_ls or []
        print(f"\n\nStart training : {self._model.get_model_config()}\n\n")

        lr_curve_visualizer = LRCurveVisualizer(
            metric_name=metric_for_graph.__class__.__name__ if metric_for_graph else "",
            calc_metric_per_epoch=calc_metrics_per_epoch)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._model.get_optimizer(),
                                                         lr_lambda=lr_scheduler_func) if lr_scheduler_func else None
        for epoch in range(epochs):
            start = time.time()
            mean_loss = self.train_epoch(data_loader)
            lr_scheduler.step(epoch) if lr_scheduler else None
            elapsed_time = time.time() - start
            lr_curve_visualizer.add(loss=mean_loss)
            # 指標算出
            metric_str = ""
            if (epoch + 1) % calc_metrics_per_epoch == 0:
                metric_str = "\n" + self.calc_metric_ls(test_dataloader, metric_ls, metric_for_graph)
                metric_val_for_graph = metric_for_graph.calc_summary() if metric_for_graph else None
                lr_curve_visualizer.add_metric(metric_val_for_graph)
                lr_curve_visualizer.save_graph_image(lr_graph_filepath)

            print(f"epoch {epoch + 1} / {epochs} :  {elapsed_time}s   --- loss: {mean_loss}{metric_str}")
            for callback in callbacks:
                callback(epoch)

    def train_epoch(self, data_loader: DataLoader) -> float:
        loss_list = []
        for train_x, teacher in data_loader:
            train_x = try_cuda(train_x)
            teacher = try_cuda(teacher)
            loss = self._model.train_batch(train_x, teacher)
            loss_list.append(loss)
        return mean(loss_list)

    def calc_metric_ls(self, data_loader: DataLoader, metric_func_ls: List[Metrics], metric_for_graph: Metrics) -> str:
        for metric_func in metric_func_ls:
            metric_func.clear()
        if metric_for_graph:
            metric_for_graph.clear()
        for x, teacher in data_loader:
            if isinstance(teacher, torch.Tensor):
                teacher = teacher.cpu().numpy()
            result = self._model.predict(x)
            for metric_func in metric_func_ls:
                metric_func.calc_one_batch(result, teacher)
            if metric_for_graph:
                metric_for_graph.calc_one_batch(result, teacher)
        metrics_str = ""
        for metric_func in metric_func_ls:
            metrics_str += f"{metric_func.__class__.__name__}: {metric_func.calc_summary()}\n\n"
        return metrics_str
