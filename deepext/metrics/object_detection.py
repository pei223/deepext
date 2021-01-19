from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch

from .base_metric import BaseMetric
from .metric_keys import DetailMetricKey, MainMetricKey


def calc_overlap_union_iou(pred: np.ndarray or None, teacher: np.ndarray) -> Tuple[float, float, float]:
    """
    :param pred: ndarray (4, )
    :param teacher: ndarray (4, )
    :return: overlap, union, iou
    """
    teacher_area = (teacher[2] - teacher[0]) * (teacher[3] - teacher[1])
    if pred is None:
        return 0.0, teacher_area, 0.0

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

    intersection_width = np.maximum(np.minimum(pred[2], teacher[2]) - np.maximum(pred[0], teacher[0]), 0)
    intersection_height = np.maximum(np.minimum(pred[3], teacher[3]) - np.maximum(pred[1], teacher[1]), 0)

    overlap = intersection_width * intersection_height
    union = teacher_area + pred_area - overlap
    iou = overlap / union
    return overlap, union, iou


class DetectionIoUByClasses(BaseMetric):
    def __init__(self, label_names: List[str], val_key: DetailMetricKey = DetailMetricKey.KEY_AVERAGE):
        assert val_key in [DetailMetricKey.KEY_TOTAL, DetailMetricKey.KEY_AVERAGE]
        self.label_names = label_names
        self.union_by_classes = [0 for _ in range(len(self.label_names))]
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self._val_key = val_key

    def clone_empty(self) -> 'DetectionIoUByClasses':
        return DetectionIoUByClasses(self.label_names.copy(), self._val_key)

    def clone(self) -> 'DetectionIoUByClasses':
        new_metric = self.clone_empty()
        new_metric.union_by_classes = self.union_by_classes.copy()
        new_metric.overlap_by_classes = self.overlap_by_classes.copy()
        return new_metric

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param teacher: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        # 全探索だと遅いのでインデックスごとにまとめておく これをRecallAndPrecisionにも
        batch_pred_by_class = []
        for pred_bboxes in pred:
            pred_bboxes_by_class = [[] for _ in range(len(self.label_names))]
            for pred_bbox in pred_bboxes:
                pred_bboxes_by_class[int(pred_bbox[-1])].append(pred_bbox)
            batch_pred_by_class.append(pred_bboxes_by_class)

        for i in range(teacher.shape[0]):
            bbox_annotations = teacher[i, :, :]
            bbox_annotations = bbox_annotations[bbox_annotations[:, -1] >= 0]

            pred_bboxes_by_class = batch_pred_by_class[i]

            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[-1])
                if pred[i] is None:
                    overlap, union, _ = calc_overlap_union_iou(None, bbox_annotation)
                    self.union_by_classes[label] += union
                    self.overlap_by_classes[label] += overlap
                    continue
                # 教師bboxに対して当てはまりの一番良いbboxを探索
                max_iou = 0
                best_union, best_overlap = 0, 0
                for pred_bbox in pred_bboxes_by_class[label]:
                    overlap, union, iou = calc_overlap_union_iou(pred_bbox, bbox_annotation)
                    if max_iou < iou:
                        max_iou = iou
                        best_union, best_overlap = union, overlap
                if max_iou <= 0:
                    overlap, union, _ = calc_overlap_union_iou(None, bbox_annotation)
                    self.union_by_classes[label] += union
                    self.overlap_by_classes[label] += overlap
                    continue
                self.union_by_classes[label] += best_union
                self.overlap_by_classes[label] += best_overlap

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
        result[DetailMetricKey.KEY_TOTAL.value] = total_overlap / total_union if total_union > 0 else 0
        return result[self._val_key.value], result

    def clear(self):
        self.union_by_classes = [0 for i in range(len(self.label_names))]
        self.overlap_by_classes = [0 for i in range(len(self.label_names))]

    def __add__(self, other: 'DetectionIoUByClasses') -> 'DetectionIoUByClasses':
        if not isinstance(other, DetectionIoUByClasses):
            raise RuntimeError(f"Bad class type. expected: {DetectionIoUByClasses.__name__}")
        if len(self.label_names) != len(other.label_names):
            raise RuntimeError(
                f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
        new_metric = self.clone_empty()
        for i in range(len(self.union_by_classes)):
            new_metric.union_by_classes[i] = self.union_by_classes[i] + other.union_by_classes[i]
            new_metric.overlap_by_classes[i] = self.overlap_by_classes[i] + other.overlap_by_classes[i]
        return new_metric

    def __truediv__(self, num: int) -> 'DetectionIoUByClasses':
        return self.clone()


class RecallAndPrecision(BaseMetric):
    def __init__(self, label_names: List[str], main_val_key: MainMetricKey = MainMetricKey.KEY_F_SCORE,
                 sub_val_key: DetailMetricKey = DetailMetricKey.KEY_TOTAL):
        self.label_names = label_names
        self.tp_by_classes = [0 for _ in range(len(self.label_names))]
        self.fp_by_classes = [0 for _ in range(len(self.label_names))]
        self.fn_by_classes = [0 for _ in range(len(self.label_names))]
        self._main_val_key = main_val_key
        self._sub_val_key = sub_val_key
        assert self._main_val_key in [MainMetricKey.KEY_RECALL, MainMetricKey.KEY_PRECISION,
                                      MainMetricKey.KEY_F_SCORE]
        assert self._sub_val_key in [DetailMetricKey.KEY_AVERAGE, DetailMetricKey.KEY_TOTAL]

    def clone_empty(self) -> 'RecallAndPrecision':
        return RecallAndPrecision(self.label_names.copy(), self._main_val_key, self._sub_val_key)

    def clone(self) -> 'RecallAndPrecision':
        new_metric = self.clone_empty()
        new_metric.tp_by_classes = self.tp_by_classes.copy()
        new_metric.fp_by_classes = self.fp_by_classes.copy()
        new_metric.fn_by_classes = self.fn_by_classes.copy()
        return new_metric

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param teacher: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        for i in range(teacher.shape[0]):
            bbox_annotations = teacher[i, :, :]
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]
            pred_bboxes = pred[i].copy()
            searched_index_ls = []
            # 全探索だと遅いのでインデックスごとにまとめておく
            idx_ls_by_classes = [[] for _ in range(len(self.label_names))]
            for j, pred_bbox in enumerate(pred_bboxes):
                idx_ls_by_classes[int(pred_bbox[-1])].append(j)

            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[-1])

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    self.fn_by_classes[label] += 1
                    continue

                # 教師bboxに当てはまっているbboxを探索
                is_matched = False
                for pred_bbox_idx in idx_ls_by_classes[label]:
                    if pred_bbox_idx in searched_index_ls:
                        continue
                    pred_bbox = pred_bboxes[pred_bbox_idx]
                    overlap, union, iou = calc_overlap_union_iou(pred_bbox, bbox_annotation)
                    if iou >= 0.5:
                        self.tp_by_classes[label] += 1
                        is_matched = True
                        searched_index_ls.append(i)
                        break
                if not is_matched:
                    self.fn_by_classes[label] += 1
            # 誤検出した数を算出
            for i, pred_bbox in enumerate(pred_bboxes):
                if i in searched_index_ls:
                    continue
                self.fp_by_classes[int(pred_bbox[-1])] += 1

    def calc_summary(self) -> Tuple[float, Dict[str, float]]:
        recall_dict, precision_dict, f_score_dict = OrderedDict(), OrderedDict(), OrderedDict()
        for label, label_name in enumerate(self.label_names):
            tp, fp, fn = self.tp_by_classes[label], self.fp_by_classes[label], self.fn_by_classes[label]
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            f_score = (2 * recall * precision) / (recall + precision) if recall + precision > 0 else 0
            recall_dict[label_name] = recall
            precision_dict[label_name] = precision
            f_score_dict[label_name] = f_score

        recall_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(recall_dict.values()) / len(self.label_names)
        precision_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(precision_dict.values()) / len(self.label_names)
        f_score_dict[DetailMetricKey.KEY_AVERAGE.value] = sum(f_score_dict.values()) / len(self.label_names)

        total_tp, total_fn, total_fp = sum(self.tp_by_classes), sum(self.fn_by_classes), sum(self.fp_by_classes)
        total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
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
        self.tp_by_classes = [0 for i in range(len(self.label_names))]
        self.fp_by_classes = [0 for i in range(len(self.label_names))]
        self.fn_by_classes = [0 for i in range(len(self.label_names))]

    def __add__(self, other: 'RecallAndPrecision') -> 'RecallAndPrecision':
        if not isinstance(other, RecallAndPrecision):
            raise RuntimeError(f"Bad class type. expected: {RecallAndPrecision.__name__}")
        if len(self.label_names) != len(other.label_names):
            raise RuntimeError(
                f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
        new_metric = self.clone_empty()
        for i in range(len(self.tp_by_classes)):
            new_metric.tp_by_classes[i] = self.tp_by_classes[i] + other.tp_by_classes[i]
            new_metric.fp_by_classes[i] = self.fp_by_classes[i] + other.fp_by_classes[i]
            new_metric.fn_by_classes[i] = self.fn_by_classes[i] + other.fn_by_classes[i]
        return new_metric

    def __truediv__(self, num: int) -> 'RecallAndPrecision':
        return self.clone()


# TODO 実装途中
class mAPByClasses:
    def __init__(self, n_classes: int):
        self._n_classes = n_classes

    def __call__(self, results, teachers):
        average_precisions = [_ for _ in range(self._n_classes)]
        for label in range(self._n_classes):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0
            for i in range(len(results)):
                detected_labels = []
                detections_by_class = results[i][label]
                annotations_by_class = teachers[i][label]
                num_annotations += annotations_by_class.shape[0]

                for detection in detections_by_class:
                    scores = np.append(scores, detection[4])

                    if annotations_by_class.shape[0] == 0:  # False detection
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = calc_bbox_overlap(np.expand_dims(detection, axis=0), annotations_by_class)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlaps = overlaps[0, assigned_annotation]

                    if assigned_annotation not in detected_labels:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_labels.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            # false_positives = false_positives[indices]
            # true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations

        return average_precisions
