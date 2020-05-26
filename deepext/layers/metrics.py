from collections import OrderedDict
from typing import List

import torch
import numpy as np
from sklearn.metrics import accuracy_score

from ..base.metrics import Metrics

SMOOTH = 1e-6


class SegmentationAccuracy(Metrics):
    def __init__(self):
        self.accuracies = []

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        pred = torch.from_numpy(pred)
        if isinstance(teacher, np.ndarray):
            teacher = torch.from_numpy(teacher)
        assert pred.shape[-1] == teacher.shape[-1] and pred.shape[-2] == teacher.shape[-2], "教師データと推論結果のサイズは同じにしてください"
        pred = pred.argmax(dim=1)
        teacher = teacher.argmax(dim=1)
        total = pred.shape[-1] * pred.shape[-2]
        pred = pred.view(pred.shape[0], -1)
        teacher = teacher.view(teacher.shape[0], -1)
        accs = (pred == teacher).sum(-1).float() / total
        self.accuracies.extend(accs.tolist())

    def calc_summary(self) -> any:
        return sum(self.accuracies) / len(self.accuracies)

    def clear(self):
        self.accuracies = []

    def name(self):
        return "segmentation accuracy"


class SegmentationAccuracyByClasses(Metrics):
    def __init__(self, label_names: List[str]):
        self.label_names = ["background", ] + label_names
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]

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
        for i, label_name in enumerate(self.label_names):
            correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
            result[label_name] = correct / (correct + incorrect)
            total_correct += correct
            total_incorrect += incorrect
        result["total"] = total_correct / (total_correct + total_incorrect)
        return result.items()

    def clear(self):
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]

    def name(self):
        return "Segmentation accuracy by classes"


class SegmentationIoUByClasses(Metrics):
    def __init__(self, label_names: List[str]):
        self.label_names = ["background", ] + label_names
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self.union_by_classes = [0 for _ in range(len(self.label_names))]

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
                union = label_teacher.shape[0] + label_pred.shape[0] - overlap
                self.overlap_by_classes[label] += overlap
                self.union_by_classes[label] += union

    def calc_summary(self):
        result = OrderedDict()
        total_overlap, total_union = 0, 0
        for i, label_name in enumerate(self.label_names):
            overlap, union = self.overlap_by_classes[i], self.union_by_classes[i]
            result[label_name] = overlap / union
            total_overlap += overlap
            total_union += union
        result["total"] = total_overlap / total_union
        return result.items()

    def clear(self):
        self.overlap_by_classes = [0 for _ in range(len(self.label_names))]
        self.union_by_classes = [0 for _ in range(len(self.label_names))]

    def name(self):
        return "Segmentation IoU by classes"


class ClassificationAccuracy(Metrics):
    def __init__(self):
        self.batch_accuracies = []

    def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
        assert pred.ndim == 1 or pred.ndim == 2
        assert teacher.ndim == 1 or teacher.ndim == 2
        if isinstance(teacher, torch.Tensor):
            teacher = teacher.cpu().numpy()
        if pred.ndim == 2:
            pred = pred.argmax(-1)
        if teacher.ndim == 2:
            teacher = teacher.argmax(-1)
        self.batch_accuracies.append(accuracy_score(pred, teacher))

    def calc_summary(self) -> any:
        return sum(self.batch_accuracies) / len(self.batch_accuracies)

    def clear(self):
        self.batch_accuracies = []

    def name(self):
        return "classification accuracy"


class ClassificationAccuracyByClasses(Metrics):
    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]

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
        for i, label_name in enumerate(self.label_names):
            correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
            result[label_name] = correct / (correct + incorrect)
            total_correct += correct
            total_incorrect += incorrect
        result["total"] = total_correct / (total_correct + total_incorrect)
        return result.items()

    def clear(self):
        self.correct_by_classes = [0 for _ in range(len(self.label_names))]
        self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]

    def name(self) -> str:
        return "Classification accuracy by classes"


class DetectionIoUByClasses(Metrics):
    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.union_by_classes = [0 for i in range(len(self.label_names))]
        self.overlap_by_classes = [0 for i in range(len(self.label_names))]

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
        for i, label_name in enumerate(self.label_names):
            overlap, union = self.overlap_by_classes[i], self.union_by_classes[i]
            result[label_name] = overlap / union
            total_overlap += overlap
            total_union += union
        result["total"] = total_overlap / total_union
        return result.items()

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

    def name(self):
        return "Detection IoU by classes"


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
