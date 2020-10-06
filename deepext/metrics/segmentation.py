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
                teacher_indices, pred_indices = (one_teacher == label), (one_pred == label)
                overlap_indices = teacher_indices & pred_indices
                overlap = np.count_nonzero(overlap_indices)
                union = np.count_nonzero(teacher_indices) + np.count_nonzero(pred_indices) - overlap
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


class SegmentationRecallPrecision(BaseMetrics):
    def __init__(self, label_names: List[str], main_val_key: MetricKey = None, sub_val_key: MetricKey = None):
        self.label_names = [MetricKey.KEY_BACKGROUND.value, ] + label_names
        self.tp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fn_by_classes = [0 for _ in range(len(self.label_names))]
        self._main_val_key = main_val_key
        self._sub_val_key = sub_val_key
        assert self._main_val_key is None or self._main_val_key in [MetricKey.KEY_RECALL, MetricKey.KEY_PRECISION,
                                                                    MetricKey.KEY_F_SCORE]
        assert self._sub_val_key is None or self._sub_val_key in [MetricKey.KEY_AVERAGE, MetricKey.KEY_TOTAL]

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
                teacher_indices, pred_indices = (one_teacher == label), (one_pred == label)
                tp = np.count_nonzero(teacher_indices & pred_indices)
                tp_fp = np.count_nonzero(pred_indices)
                tp_fn = np.count_nonzero(teacher_indices)
                self.tp_by_classes[label] += tp
                self.tp_fp_by_classes[label] += tp_fp
                self.tp_fn_by_classes[label] += tp_fn

    def calc_summary(self) -> any:
        recall_dict, precision_dict, f_score_dict = OrderedDict(), OrderedDict(), OrderedDict()
        total_tp, total_tp_fp, total_tp_fn = 0, 0, 0
        for i, label_name in enumerate(self.label_names):
            tp, tp_fp, tp_fn = self.tp_by_classes[i], self.tp_fp_by_classes[i], self.tp_fn_by_classes[i]
            recall = tp / tp_fn if tp_fn > 0 else 0
            precision = tp / tp_fp if tp_fp > 0 else 0
            f_score = (2 * recall * precision) / (recall + precision) if recall + precision > 0 else 0

            recall_dict[label_name] = recall
            precision_dict[label_name] = precision
            f_score_dict[label_name] = f_score

            total_tp += tp
            total_tp_fp += tp_fp
            total_tp_fn += tp_fn

        recall_dict[MetricKey.KEY_AVERAGE.value] = sum(recall_dict.values()) / len(self.label_names)
        precision_dict[MetricKey.KEY_AVERAGE.value] = sum(precision_dict.values()) / len(self.label_names)
        f_score_dict[MetricKey.KEY_AVERAGE.value] = sum(f_score_dict.values()) / len(self.label_names)

        total_recall = total_tp / (total_tp_fn) if total_tp_fn > 0 else 0
        total_precision = total_tp / (total_tp_fp) if total_tp_fp > 0 else 0
        total_f_score = (2 * total_recall * total_precision) / (
                total_recall + total_precision) if total_recall + total_precision > 0 else 0

        recall_dict[MetricKey.KEY_TOTAL.value] = total_recall
        precision_dict[MetricKey.KEY_TOTAL.value] = total_precision
        f_score_dict[MetricKey.KEY_TOTAL.value] = total_f_score

        result = {
            MetricKey.KEY_RECALL.value: recall_dict,
            MetricKey.KEY_PRECISION.value: precision_dict,
            MetricKey.KEY_F_SCORE.value: f_score_dict
        }
        if self._main_val_key:
            if self._sub_val_key:
                return result[self._main_val_key.value][self._sub_val_key.value]
            return result[self._main_val_key.value.value]
        return result

    def clear(self):
        self.tp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fn_by_classes = [0 for _ in range(len(self.label_names))]
