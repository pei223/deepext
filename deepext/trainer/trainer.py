from typing import List, Callable

from torch.utils.data import DataLoader
import torch
from statistics import mean
import time
import tqdm

from ..models.base import BaseModel
from ..utils.tensor_util import try_cuda
from ..metrics.base_metrics import BaseMetrics
from .learning_curve_visualizer import LearningCurveVisualizer
from .progress_writer import ProgressWriter, StdOutProgressWriter


class Trainer:
    def __init__(self, model: BaseModel, learning_curve_visualizer: LearningCurveVisualizer = None,
                 progress_writer: ProgressWriter = None):
        """
        :param model:
        :param learning_curve_visualizer: Write learning curve graph.
        :param progress_writer: Write output.
        """
        self._model: BaseModel = model
        self._writer = progress_writer or StdOutProgressWriter()
        self._visualizer = learning_curve_visualizer

    def fit(self, train_data_loader: DataLoader, epochs: int, test_data_loader: DataLoader = None,
            callbacks: List[Callable[[int, ], None]] = None, metric_ls: List[BaseMetrics] = None,
            metric_for_graph: BaseMetrics = None, lr_scheduler_func: Callable[[int, ], float] = None,
            calc_metrics_per_epoch: int = 5):
        """
        :param train_data_loader: DataLoader for training
        :param epochs:
        :param test_data_loader: DataLoader for test
        :param callbacks:
        :param metric_ls: 指標リスト
        :param metric_for_graph:
        :param lr_scheduler_func: 学習率スケジューリング関数
        :param calc_metrics_per_epoch: 何エポックごとに指標を計算するか
        """
        callbacks, metric_ls = callbacks or [], metric_ls or []
        self._writer.out_training_start(str(self._model.get_model_config()))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._model.get_optimizer(),
                                                         lr_lambda=lr_scheduler_func) if lr_scheduler_func else None
        for epoch in range(epochs):
            start = time.time()
            mean_loss = self.train_epoch(train_data_loader)
            lr_scheduler.step(epoch + 1) if lr_scheduler else None
            self._visualizer.add_loss(mean_loss) if self._visualizer else None
            self._writer.out_epoch(epoch, max_epoch=epochs, elapsed_time=time.time() - start, mean_loss=mean_loss)
            for callback in callbacks:
                callback(epoch)

            if (epoch + 1) % calc_metrics_per_epoch != 0 or len(metric_ls) == 0:
                continue

            # 指標算出
            # print("\nTrain Metrics\n" + self.calc_metrics(data_loader, metric_ls, metric_for_graph))
            # train_metric_val_for_graph = metric_for_graph.calc_summary() if metric_for_graph else None
            self._writer.out_heading("Test Metrics")
            # self.calc_metrics(data_loader, metric_ls, metric_for_graph, mode="train")
            self.calc_metrics(test_data_loader, metric_ls, metric_for_graph, mode="test")
            if self._visualizer is None:
                continue
            metric_test_val_for_graph = metric_for_graph.calc_summary()[0]
            self._visualizer.add_metrics(test_metric=metric_test_val_for_graph,
                                         train_metric=None,
                                         # train_metric=train_metric_val_for_graph,
                                         calc_metric_per_epoch=calc_metrics_per_epoch)
            self._visualizer.save_graph_image()

    def train_epoch(self, data_loader: DataLoader) -> float:
        loss_list = []
        for train_x, teacher in tqdm.tqdm(data_loader):
            train_x, teacher = try_cuda(train_x), try_cuda(teacher)
            if train_x.shape[0] == 1:  # Batch normalizationが動かなくなるため
                continue
            loss = self._model.train_batch(train_x, teacher)
            loss_list.append(loss)
        return mean(loss_list)

    def calc_metrics(self, data_loader: DataLoader, metric_func_ls: List[BaseMetrics],
                     metric_for_graph: BaseMetrics or None, mode: str):
        start = time.time()
        for metric_func in metric_func_ls:
            metric_func.clear()
        metric_for_graph.clear() if metric_for_graph else None
        for x, teacher in data_loader:
            if isinstance(teacher, torch.Tensor):
                teacher = teacher.cpu().numpy()
            result = self._model.predict(try_cuda(x))
            for metric_func in metric_func_ls:
                metric_func.calc_one_batch(result, teacher)
            if metric_for_graph:
                metric_for_graph.calc_one_batch(result, teacher)
        for metric_func in metric_func_ls:
            self._writer.out_metrics(metric_func.__class__.__name__, str(metric_func.calc_summary()[1]), mode) \
                .out_small_divider()
        self._writer.out_elapsed_time(time.time() - start).out_divider()
