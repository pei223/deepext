from collections import OrderedDict
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from deepext.base import Metrics
from .keys import MetricKey


class ClassificationAccuracyByClasses(Metrics):
    def __init__(self, label_names: List[str], val_key: MetricKey = None):
        self.label_names = label_names
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
        self._val_key = val_key

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        assert pred.ndim == 1 or pred.ndim == 2
        assert teacher.ndim == 1 or teacher.ndim == 2
        if isinstance(teacher, torch.Tensor):
            teacher = teacher.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if pred.ndim == 2:
            pred = pred.argmax(-1)
        if teacher.ndim == 2:
            teacher = teacher.argmax(-1)
        result_flags: np.ndarray = (pred == teacher)
        for label in range(len(self.label_names)):
            label_index = (teacher == label)
            class_result_flags = result_flags[label_index]
            correct = np.count_nonzero(class_result_flags)
            incorrect = class_result_flags.shape[0] - correct
            self.correct_by_classes[label] += correct
            self.incorrect_by_classes[label] += incorrect

    def calc_summary(self) -> any:
        result = OrderedDict()
        total_correct, total_incorrect = 0, 0
        avg_acc = 0.0
        for i, label_name in enumerate(self.label_names):
            correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
            result[label_name] = correct / (correct + incorrect)
            total_correct += correct
            total_incorrect += incorrect
            avg_acc += result[label_name]
        result[MetricKey.KEY_TOTAL.value] = total_correct / (total_correct + total_incorrect)
        result[MetricKey.KEY_AVERAGE.value] = avg_acc / len(self.label_names)
        if self._val_key:
            return result[self._val_key.value]
        return list(result.items())

    def clear(self):
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
