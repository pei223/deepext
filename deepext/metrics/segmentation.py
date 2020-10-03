from collections import OrderedDict
from typing import List

import numpy as np
import torch

from .base_metrics import BaseMetrics
from .keys import MetricKey


class SegmentationAccuracyByClasses(BaseMetrics):
    def __init__(self, label_names: List[str], val_key: MetricKey = None):
        assert val_key is None or val_key in [MetricKey.KEY_TOTAL, MetricKey.KEY_AVERAGE,
                                              MetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND]
        self.label_names = [MetricKey.KEY_BACKGROUND.value, ] + label_names
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
        self._val_key = val_key

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, classes, height, width)
        :param teacher: (Batch size, classes, height, width)
        """
        pred = torch.from_numpy(pred)
        if isinstance(teacher, torch.Tensor):
            teacher = teacher.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        assert pred.ndim == 4 and teacher.ndim == 4
        assert pred.shape[-1] == teacher.shape[-1] and pred.shape[-2] == teacher.shape[-2], "教師データと推論結果のサイズは同じにしてください"

        pred = pred.argmax(1).reshape([pred.shape[0], -1])
        teacher = teacher.argmax(1).reshape(teacher.shape[0], -1)
        # NOTE バッチまとめて計算するとメモリが大きくなりnumpy演算が正常にできないため、1データごとに処理
        for batch_i in range(pred.shape[0]):
            one_pred, one_teacher = pred[batch_i], teacher[batch_i]
            for label in range(len(self.label_names)):
                label_index = (one_teacher == label)
                label_result_flags = (one_teacher[label_index] == one_pred[label_index])
                correct = np.count_nonzero(label_result_flags)
                incorrect = label_result_flags.shape[0] - correct
                self.correct_by_classes[label] += correct
                self.incorrect_by_classes[label] += incorrect

    def calc_summary(self):
        result = OrderedDict()
        total_correct, total_incorrect = 0, 0
        avg_acc = 0.0
        for i, label_name in enumerate(self.label_names):
            correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
            result[label_name] = correct / (correct + incorrect) if correct + incorrect > 0 else 0
            total_correct += correct
            total_incorrect += incorrect
            avg_acc += result[label_name]
        result[MetricKey.KEY_AVERAGE.value] = avg_acc / len(self.label_names)
        result[MetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND.value] = (avg_acc - result[MetricKey.KEY_BACKGROUND.value]) / (
                len(self.label_names) - 1)
        result[MetricKey.KEY_TOTAL.value] = total_correct / (total_correct + total_incorrect)
        if self._val_key:
            return result[self._val_key.value]
        return list(result.items())

    def clear(self):
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]


class SegmentationIoUByClasses(BaseMetrics):
    def __init__(self, label_names: List[str], val_key: MetricKey = None):
        assert val_key is None or val_key in [MetricKey.KEY_TOTAL, MetricKey.KEY_AVERAGE,
                                              MetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND]
        self.label_names = [MetricKey.KEY_BACKGROUND.value, ] + label_names
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self.union_by_classes = [0 for _ in range(len(self.label_names))]
        self._val_key = val_key

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, classes, height, width)
        :param teacher: (Batch size, classes, height, width)
        """
        pred = torch.from_numpy(pred)
        if isinstance(teacher, torch.Tensor):
            teacher = teacher.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        assert pred.ndim == 4 and teacher.ndim == 4
        assert pred.shape[-1] == teacher.shape[-1] and pred.shape[-2] == teacher.shape[-2], "教師データと推論結果のサイズは同じにしてください"

        pred = pred.argmax(1).reshape([pred.shape[0], -1])
        teacher = teacher.argmax(1).reshape(teacher.shape[0], -1)
        # NOTE バッチまとめて計算するとメモリが大きくなりnumpy演算が正常にできないため、1データごとに処理
        for batch_i in range(pred.shape[0]):
            one_pred, one_teacher = pred[batch_i], teacher[batch_i]
            for label in range(len(self.label_names)):
                label_teacher, label_pred = (one_teacher == label), (one_pred == label)
                label_result_flags = (one_teacher[label_teacher] == one_pred[label_teacher])
                overlap = np.count_nonzero(label_result_flags)
                union = np.count_nonzero(label_teacher) + np.count_nonzero(label_pred) - overlap
                self.overlap_by_classes[label] += overlap
                self.union_by_classes[label] += union

    def calc_summary(self):
        result = OrderedDict()
        total_overlap, total_union = 0, 0
        avg_iou = 0.0
        for i, label_name in enumerate(self.label_names):
            overlap, union = self.overlap_by_classes[i], self.union_by_classes[i]
            result[label_name] = overlap / union if union > 0 else 0
            total_overlap += overlap
            total_union += union
            avg_iou += result[label_name]
        result[MetricKey.KEY_AVERAGE.value] = avg_iou / len(self.label_names)
        result[MetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND.value] = (avg_iou - result[MetricKey.KEY_BACKGROUND.value]) / (
                len(self.label_names) - 1)
        result[MetricKey.KEY_TOTAL.value] = total_overlap / total_union if total_union > 0 else 0
        if self._val_key:
            return result[self._val_key.value]
        return list(result.items())

    def clear(self):
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self.union_by_classes = [0 for _ in range(len(self.label_names))]

# TODO Recall/Precision/F Score
