from torch.utils.data import DataLoader
from typing import List, Callable
from statistics import mean
import numpy as np
import time

from deepext.base import BaseModel
from deepext.utils.tensor_util import try_cuda
from ..base import Metrics
import torch


class Trainer:
    def __init__(self, model: BaseModel):
        self._model: BaseModel = model

    def fit(self, data_loader: DataLoader, epochs: int, test_dataloader: DataLoader = None,
            callbacks: List[Callable[[int, ], None]] = None, metric_func_ls: List[Metrics] = None,
            lr_scheduler_func: Callable[[int, ], float] = None, calc_metrics_per_epoch: int = 5):
        callbacks, metric_func_ls = callbacks or [], metric_func_ls or []
        print(f"\n\nStart training : {self._model.get_model_config()}\n\n")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._model.get_optimizer(),
                                                         lr_lambda=lr_scheduler_func) if lr_scheduler_func else None
        for epoch in range(epochs):
            start = time.time()
            mean_loss = self.train_epoch(data_loader)
            lr_scheduler.step(epoch) if lr_scheduler else None
            elapsed_time = time.time() - start
            metric_str = ""
            if (epoch + 1) % calc_metrics_per_epoch == 0:
                metric_str = "\n" + self.calc_metric_ls(test_dataloader, metric_func_ls)
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

    def calc_metric_ls(self, data_loader: DataLoader, metric_func_ls: List[Metrics]) -> str:
        for metric_func in metric_func_ls:
            metric_func.clear()
        for x, teacher in data_loader:
            if isinstance(teacher, torch.Tensor):
                teacher = teacher.cpu().numpy()
            result = self._model.predict(x)
            for metric_func in metric_func_ls:
                metric_func.calc_one_batch(result, teacher)
        metrics_str = ""
        for metric_func in metric_func_ls:
            metrics_str += f"{metric_func.name()}: {metric_func.calc_summary()}"
        return metrics_str
