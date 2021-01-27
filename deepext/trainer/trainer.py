from typing import List, Callable

from torch.utils.data import DataLoader
import torch
from statistics import mean
import time
import tqdm

from .callbacks import ModelCallback
from ..models.base import BaseModel
from ..utils.tensor_util import try_cuda
from ..metrics.base_metric import BaseMetric
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
            callbacks: List[ModelCallback] = None, metric_ls: List[BaseMetric] = None,
            metric_for_graph: BaseMetric = None,
            epoch_lr_scheduler_func: Callable[[int, ], float] = None,
            loss_lr_scheduler=None,
            calc_metrics_per_epoch: int = 5, required_train_metric=False):
        """
        :param train_data_loader: DataLoader for training
        :param epochs:
        :param test_data_loader: DataLoader for test
        :param callbacks:
        :param metric_ls: 指標リスト
        :param metric_for_graph:
        :param epoch_lr_scheduler_func: エポックを引数にした学習率スケジューリング関数
        :param loss_lr_scheduler: Lossを引数にした学習率スケジューリング
        :param calc_metrics_per_epoch: 何エポックごとに指標を計算するか
        :param required_train_metric: 訓練データの指標も出力するかどうか
        """
        if loss_lr_scheduler is not None and epoch_lr_scheduler_func is not None:
            raise ValueError("Learning rate scheduler must be one.")
        if loss_lr_scheduler is not None and not hasattr(loss_lr_scheduler, "step"):
            raise AttributeError("loss_lr_scheduler is required step function.")

        callbacks, metric_ls = callbacks or [], metric_ls or []
        self._writer.out_training_start(str(self._model.get_model_config()))

        test_metric_for_graph = metric_for_graph
        train_metric_for_graph = metric_for_graph.clone_empty()

        epoch_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._model.get_optimizer(),
                                                               lr_lambda=epoch_lr_scheduler_func) \
            if epoch_lr_scheduler_func else None
        for epoch in range(epochs):
            start = time.time()
            mean_loss = self.train_epoch(train_data_loader)
            loss_lr_scheduler.step(mean_loss) if loss_lr_scheduler else None
            epoch_lr_scheduler.step(epoch + 1) if epoch_lr_scheduler else None
            self._visualizer.add_loss(mean_loss) if self._visualizer else None
            self._writer.out_epoch(epoch, max_epoch=epochs, elapsed_time=time.time() - start, mean_loss=mean_loss)
            for callback in callbacks:
                callback(epoch)

            if (epoch + 1) % calc_metrics_per_epoch != 0 or len(metric_ls) == 0:
                continue

            # 指標算出
            if required_train_metric:
                self._writer.out_heading("Train Metrics")
                self.calc_metrics(train_data_loader, metric_ls, train_metric_for_graph, mode="train")
            else:
                self.calc_metrics(train_data_loader, [], train_metric_for_graph, mode="train")

            self._writer.out_heading("Test Metrics")
            self.calc_metrics(test_data_loader, metric_ls, test_metric_for_graph, mode="test")
            if self._visualizer is None:
                continue
            test_metric_val_for_graph = test_metric_for_graph.calc_summary()[0]
            train_metric_val_for_graph = train_metric_for_graph.calc_summary()[0]
            self._visualizer.add_metrics(test_metric=test_metric_val_for_graph,
                                         train_metric=train_metric_val_for_graph,
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

    def calc_metrics(self, data_loader: DataLoader, metric_func_ls: List[BaseMetric],
                     metric_for_graph: BaseMetric or None, mode: str):
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
