from typing import List, Callable

from torch.utils.data import DataLoader
import torch
from statistics import mean
import numpy as np
import time

from ..models.base import BaseModel
from ..utils.tensor_util import try_cuda
from ..metrics.base_metrics import BaseMetrics
from .learning_curve_visualizer import LearningCurveVisualizer


class Trainer:
    def __init__(self, model: BaseModel):
        self._model: BaseModel = model

    def fit(self, data_loader: DataLoader, epochs: int, test_dataloader: DataLoader = None,
            callbacks: List[Callable[[int, ], None]] = None, metric_ls: List[BaseMetrics] = None,
            lr_scheduler_func: Callable[[int, ], float] = None, calc_metrics_per_epoch: int = 5,
            learning_curve_visualizer: LearningCurveVisualizer = None):
        """
        :param data_loader: DataLoader for training
        :param epochs:
        :param test_dataloader: DataLoader for test
        :param callbacks:
        :param metric_ls: 指標リスト
        :param lr_scheduler_func: 学習率スケジューリング関数
        :param calc_metrics_per_epoch: 何エポックごとに指標を計算するか
        :param learning_curve_visualizer: 学習曲線グラフ可視化.
        """
        callbacks, metric_ls = callbacks or [], metric_ls or []
        print(f"\n\nStart training : {self._model.get_model_config()}\n\n")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._model.get_optimizer(),
                                                         lr_lambda=lr_scheduler_func) if lr_scheduler_func else None
        for epoch in range(epochs):
            start = time.time()
            mean_loss = self.train_epoch(data_loader)
            lr_scheduler.step(epoch) if lr_scheduler else None
            elapsed_time = time.time() - start
            if learning_curve_visualizer:
                learning_curve_visualizer.add_loss(mean_loss)
            print(f"epoch {epoch + 1} / {epochs} :  {elapsed_time}s   --- loss: {mean_loss}")
            for callback in callbacks:
                callback(epoch)
            # 指標算出
            if (epoch + 1) % calc_metrics_per_epoch == 0:
                metric_for_graph = learning_curve_visualizer.metric_for_graph if learning_curve_visualizer else None
                # print("\nTrain Metrics\n" + self.calc_metrics(data_loader, metric_ls, metric_for_graph))
                # train_metric_val_for_graph = metric_for_graph.calc_summary() if metric_for_graph else None
                print("\nTest Metrics \n" + self.calc_metrics(test_dataloader, metric_ls, metric_for_graph))
                if learning_curve_visualizer:
                    metric_test_val_for_graph = metric_for_graph.calc_summary()
                    learning_curve_visualizer.add_metrics(test_metric=metric_test_val_for_graph,
                                                          train_metric=None,
                                                          # train_metric=train_metric_val_for_graph,
                                                          calc_metric_per_epoch=calc_metrics_per_epoch)
                    learning_curve_visualizer.save_graph_image()

    def train_epoch(self, data_loader: DataLoader) -> float:
        loss_list = []
        for train_x, teacher in data_loader:
            train_x = try_cuda(train_x)
            teacher = try_cuda(teacher)
            loss = self._model.train_batch(train_x, teacher)
            loss_list.append(loss)
        return mean(loss_list)

    def calc_metrics(self, data_loader: DataLoader, metric_func_ls: List[BaseMetrics],
                     metric_for_graph: BaseMetrics or None) -> str:
        start = time.time()
        for metric_func in metric_func_ls:
            metric_func.clear()
        if metric_for_graph:
            metric_for_graph.clear()
        for x, teacher in data_loader:
            if isinstance(teacher, torch.Tensor):
                teacher = teacher.cpu().numpy()
            result = self._model.predict(try_cuda(x))
            for metric_func in metric_func_ls:
                metric_func.calc_one_batch(result, teacher)
            if metric_for_graph:
                metric_for_graph.calc_one_batch(result, teacher)
        metrics_str = ""
        for metric_func in metric_func_ls:
            metrics_str += f"{metric_func.__class__.__name__}: {metric_func.calc_summary()}\n\n"
        metrics_str += f"Elapsed time: {time.time() - start}s\n\n\n"
        return metrics_str
