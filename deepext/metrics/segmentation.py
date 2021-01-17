from collections import OrderedDict
from typing import List, Dict, Tuple

import numpy as np
import torch

from .base_metrics import BaseMetrics
from .keys import DetailMetricKey, MainMetricKey


class SegmentationAccuracyByClasses(BaseMetrics):
    def __init__(self, label_names: List[str],
                 val_key: DetailMetricKey = DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND):
        assert val_key in [DetailMetricKey.KEY_TOTAL, DetailMetricKey.KEY_AVERAGE,
                           DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND]
        self.label_names = [DetailMetricKey.KEY_BACKGROUND.value, ] + label_names
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
        self._val_key = val_key

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, classes, height, width)
        :param teacher: (Batch size, classes, height, width)
        """
        pred = torch.from_numpy(pred)

        teacher = teacher.cpu().numpy() if isinstance(teacher, torch.Tensor) else teacher
        pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred

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

    def calc_summary(self) -> Tuple[float, Dict[str, float]]:
        result = OrderedDict()
        total_correct, total_incorrect = 0, 0
        avg_acc = 0.0
        for i, label_name in enumerate(self.label_names):
            correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
            result[label_name] = correct / (correct + incorrect) if correct + incorrect > 0 else 0
            total_correct += correct
            total_incorrect += incorrect
            avg_acc += result[label_name]
        result[DetailMetricKey.KEY_AVERAGE.value] = avg_acc / len(self.label_names)
        result[DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND.value] = (avg_acc - result[
            DetailMetricKey.KEY_BACKGROUND.value]) / (
                                                                               len(self.label_names) - 1)
        result[DetailMetricKey.KEY_TOTAL.value] = total_correct / (total_correct + total_incorrect)
        return result[self._val_key.value], result

    def clear(self):
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]

    def add(self, other: 'SegmentationAccuracyByClasses'):
        if not isinstance(other, SegmentationAccuracyByClasses):
            raise RuntimeError(f"Bad class type. expected: {SegmentationAccuracyByClasses.__name__}")
        if len(self.label_names) != len(other.label_names):
            raise RuntimeError(
                f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
        for i in range(len(self.correct_by_classes)):
            self.correct_by_classes[i] += other.correct_by_classes[i]
            self.incorrect_by_classes[i] += other.incorrect_by_classes[i]

    def div(self, num: int):
        pass


class SegmentationIoUByClasses(BaseMetrics):
    def __init__(self, label_names: List[str],
                 val_key: DetailMetricKey = DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND):
        assert val_key in [DetailMetricKey.KEY_TOTAL, DetailMetricKey.KEY_AVERAGE,
                           DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND]
        self.label_names = [DetailMetricKey.KEY_BACKGROUND.value, ] + label_names
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

    def calc_summary(self) -> Tuple[float, Dict[str, float]]:
        result = OrderedDict()
        total_overlap, total_union = 0, 0
        avg_iou = 0.0
        for i, label_name in enumerate(self.label_names):
            overlap, union = self.overlap_by_classes[i], self.union_by_classes[i]
            result[label_name] = overlap / union if union > 0 else 0
            total_overlap += overlap
            total_union += union
            avg_iou += result[label_name]
        result[DetailMetricKey.KEY_AVERAGE.value] = avg_iou / len(self.label_names)
        result[DetailMetricKey.KEY_AVERAGE_WITHOUT_BACKGROUND.value] = (avg_iou - result[
            DetailMetricKey.KEY_BACKGROUND.value]) / (
                                                                               len(self.label_names) - 1)
        result[DetailMetricKey.KEY_TOTAL.value] = total_overlap / total_union if total_union > 0 else 0
        return result[self._val_key.value], result

    def clear(self):
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self.union_by_classes = [0 for _ in range(len(self.label_names))]

    def add(self, other: 'SegmentationIoUByClasses'):
        if not isinstance(other, SegmentationIoUByClasses):
            raise RuntimeError(f"Bad class type. expected: {SegmentationIoUByClasses.__name__}")
        if len(self.label_names) != len(other.label_names):
            raise RuntimeError(
                f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
        for i in range(len(self.overlap_by_classes)):
            self.overlap_by_classes[i] += other.overlap_by_classes[i]
            self.union_by_classes[i] += other.union_by_classes[i]

    def div(self, num: int):
        pass


class SegmentationRecallPrecision(BaseMetrics):
    def __init__(self, label_names: List[str], main_val_key: MainMetricKey = MainMetricKey.KEY_F_SCORE,
                 sub_val_key: DetailMetricKey = DetailMetricKey.KEY_TOTAL):
        assert main_val_key in [MainMetricKey.KEY_RECALL, MainMetricKey.KEY_PRECISION,
                                MainMetricKey.KEY_F_SCORE]
        assert sub_val_key in [DetailMetricKey.KEY_AVERAGE, DetailMetricKey.KEY_TOTAL]
        self.label_names = [DetailMetricKey.KEY_BACKGROUND.value, ] + label_names
        self.tp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fn_by_classes = [0 for _ in range(len(self.label_names))]
        self._main_val_key = main_val_key
        self._sub_val_key = sub_val_key

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

    def calc_summary(self) -> Tuple[float, Dict[str, float]]:
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

        recall_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(recall_dict.values()) / len(self.label_names)
        precision_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(precision_dict.values()) / len(self.label_names)
        f_score_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(f_score_dict.values()) / len(self.label_names)

        total_recall = total_tp / (total_tp_fn) if total_tp_fn > 0 else 0
        total_precision = total_tp / (total_tp_fp) if total_tp_fp > 0 else 0
        total_f_score = (2 * total_recall * total_precision) / (
                total_recall + total_precision) if total_recall + total_precision > 0 else 0

        recall_dict[DetailMetricKey.KEY_TOTAL.value] = total_recall
        precision_dict[DetailMetricKey.KEY_TOTAL.value] = total_precision
        f_score_dict[DetailMetricKey.KEY_TOTAL.value] = total_f_score

        result = {
            MainMetricKey.KEY_RECALL.value: recall_dict,
            MainMetricKey.KEY_PRECISION.value: precision_dict,
            MainMetricKey.KEY_F_SCORE.value: f_score_dict
        }
        return result[self._main_val_key.value][self._sub_val_key.value], result

    def clear(self):
        self.tp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fp_by_classes = [0 for _ in range(len(self.label_names))]
        self.tp_fn_by_classes = [0 for _ in range(len(self.label_names))]

    def add(self, other: 'SegmentationRecallPrecision'):
        if not isinstance(other, SegmentationRecallPrecision):
            raise RuntimeError(f"Bad class type. expected: {SegmentationRecallPrecision.__name__}")
        if len(self.label_names) != len(other.label_names):
            raise RuntimeError(
                f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
        for i in range(len(self.tp_by_classes)):
            self.tp_by_classes[i] += other.tp_by_classes[i]
            self.tp_fp_by_classes[i] += other.tp_fp_by_classes[i]
            self.tp_fn_by_classes[i] += other.tp_fn_by_classes[i]

    def div(self, num: int):
        pass
