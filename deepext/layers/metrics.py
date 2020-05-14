import torch
import numpy as np
from sklearn.metrics import accuracy_score

SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = torch.max(outputs, 1)[0].int()
    labels = torch.max(labels, 1)[0].int()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    # print(((intersection + 0.001) / (union - intersection + 0.001)).shape)
    return (intersection + 0.001) / (union - intersection + 0.001)


def segmentation_accuracy(pred: np.ndarray, teacher: np.ndarray) -> float:
    """
    :param pred: Batch * H * W
    :param teacher: Batch * H * W
    :return:
    """
    pred = torch.from_numpy(pred)
    if isinstance(teacher, np.ndarray):
        teacher = torch.from_numpy(teacher)
    assert pred.shape[-1] == teacher.shape[-1] and pred.shape[-2] == teacher.shape[-2], "教師データと推論結果のサイズは同じにしてください"
    pred = pred.argmax(dim=1)
    teacher = teacher.argmax(dim=1)
    total = pred.shape[-1] * pred.shape[-2]
    pred = pred.view(pred.shape[0], -1)
    teacher = teacher.view(teacher.shape[0], -1)
    correct = (pred == teacher).sum(-1).float()
    return (correct / total).mean(0).item()


def segmentation_accuracy_by_classes(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    :param pred: Batch * H * W
    :param teacher: Batch * H * W
    :return: Accuracy by classes,
    """
    assert pred.shape[-1] == teacher.shape[-1] and pred.shape[-2] == teacher.shape[-2], "教師データと推論結果のサイズは同じにしてください"
    total = pred.shape[-1] * pred.shape[-2] * pred.shape[0]
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    teacher = teacher.view(teacher.shape[0], pred.shape[1], -1)
    correct = (pred == teacher).sum(-1).sum(0).float()
    return correct / total


def classification_accuracy(pred: np.ndarray, teacher: np.ndarray):
    assert pred.ndim == 1 or pred.ndim == 2
    assert teacher.ndim == 1 or teacher.ndim == 2
    if pred.ndim == 2:
        pred = pred.argmax(-1)
    if teacher.ndim == 2:
        teacher = teacher.argmax(-1)
    return accuracy_score(pred, teacher)
