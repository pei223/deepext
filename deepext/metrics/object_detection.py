from collections import OrderedDict
from typing import List

import numpy as np
import torch

from deepext.base import Metrics

from .keys import MetricKey


class DetectionIoUByClasses(Metrics):
    def __init__(self, label_names: List[str], val_key: MetricKey = None):
        self.label_names = label_names
        self.union_by_classes = [0 for i in range(len(self.label_names))]
        self.overlap_by_classes = [0 for i in range(len(self.label_names))]
        self._val_key = val_key

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        """
        :param pred: (Batch size, classes, bounding boxes by class(variable length))
        :param teacher: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        for i in range(teacher.shape[0]):
            bbox_annotations = teacher[i, :, :]
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]
            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[-1])
                if pred[i] is None or pred[i][label] is None:
                    overlap, union = self._calc_bbox_overlap_and_union(None, bbox_annotation)
                    self.union_by_classes[label] += union
                    self.overlap_by_classes[label] += overlap
                    continue
                pred_bboxes = pred[i][label]
                for pred_bbox in pred_bboxes:
                    overlap, union = self._calc_bbox_overlap_and_union(pred_bbox, bbox_annotation)
                    self.union_by_classes[label] += union
                    self.overlap_by_classes[label] += overlap

    def calc_summary(self):
        result = OrderedDict()
        total_overlap, total_union = 0, 0
        avg_iou = 0.0
        for i, label_name in enumerate(self.label_names):
            overlap, union = self.overlap_by_classes[i], self.union_by_classes[i]
            result[label_name] = overlap / union
            total_overlap += overlap
            total_union += union
            avg_iou += result[label_name]
        result[MetricKey.KEY_AVERAGE.value] = avg_iou / len(self.label_names)
        result[MetricKey.KEY_TOTAL.value] = total_overlap / total_union
        if self._val_key:
            return result[self._val_key.value]
        return list(result.items())

    def _calc_bbox_overlap_and_union(self, pred: np.ndarray or None, teacher: np.ndarray):
        """
        :param pred: ndarray (4, )
        :param teacher: ndarray (4,
        :return:
        """
        teacher_area = (teacher[2] - teacher[0]) * (teacher[3] - teacher[1])
        if pred is None:
            return 0.0, teacher_area

        pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

        intersection_width = np.maximum(np.minimum(pred[2], teacher[2]) - np.maximum(pred[0], teacher[0]), 0)
        intersection_height = np.maximum(np.minimum(pred[3], teacher[3]) - np.maximum(pred[1], teacher[1]), 0)

        overlap = intersection_width * intersection_height
        union = teacher_area + pred_area - overlap

        return overlap, union

    def clear(self):
        self.union_by_classes = [0 for i in range(len(self.label_names))]
        self.overlap_by_classes = [0 for i in range(len(self.label_names))]


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
