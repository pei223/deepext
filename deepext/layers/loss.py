import torch
from .metrics import *
from torch import nn
from torch.nn import functional as F


class IoULoss(torch.nn.Module):
    def init(self):
        super(IoULoss, self).init()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor):
        return (1 - iou_pytorch(pred, teacher)).mean()


class JaccardLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, teacher: torch.Tensor, smooth=1.0):
        teacher = teacher.float()
        intersection = (pred * teacher).sum((-1, -2))
        sum_ = (pred.abs() + pred.abs()).sum((-1, -2))
        jaccard = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jaccard).mean(1).mean(0)


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor, smooth=1.0):
        teacher = teacher.float()
        intersection = (pred * teacher).sum((-1, -2))
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[0], teacher.shape[1], -1)
        pred_sum = pred.sum((-1,))
        teacher_sum = teacher.sum((-1,))
        return (1 - ((2. * intersection + smooth) / (pred_sum + teacher_sum + smooth))).mean((1,)).mean((0,))


class DiceLossWithAreaPenalty(torch.nn.Module):
    def init(self):
        super(DiceLossWithAreaPenalty, self).init()

    def forward(self, pred, teacher, smooth=1.0):
        """
        :param pred: 推論結果 (Batch size * class num * height * width)
        :param teacher: 教師データ (Batch size * class num * height * weigh )
        :param smooth:
        :return:
        """
        intersection = (pred * teacher).sum((-1, -2)).mean((0,))
        pred = pred.contiguous().view(pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[1], -1)
        pred_sum = pred.sum((-1,))
        teacher_sum = teacher.sum((-1,))
        return (1 - ((2. * intersection + smooth) / (pred_sum + teacher_sum + smooth))).mean()

        # total = torch.sum(teacher * teacher).type('torch.DoubleTensor')
        # teacher_areas_by_class = teacher.sum(axis=[0, -1, -2]).type('torch.DoubleTensor')
        # penalty = (total - teacher_areas_by_class) / total
        #
        # intersections_by_class = (pred * teacher).sum(axis=[0, -1, -2, ])
        # pred_sum, teacher_sum = torch.sum(pred * pred), torch.sum(teacher * teacher)
        #
        # return 1 - ((2. * (intersections_by_class).sum() + smooth) / (pred_sum + teacher_sum + smooth))


class SegmentationTypedLoss(nn.Module):
    def __init__(self, loss_type: str):
        super(SegmentationTypedLoss, self).__init__()
        assert loss_type.lower() in ["ce+dice", "dice", "ce", ]
        self._loss_type = loss_type.lower()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor):
        """
        :param pred: B * C * H * W
        :param teacher: B * C * H * W
        """
        assert pred.ndim == 4 and teacher.ndim == 4
        if self._loss_type == "ce+dice":
            return AdaptiveCrossEntropyLoss()(pred, teacher) + DiceLoss()(pred, teacher)
        if self._loss_type == "ce":
            return AdaptiveCrossEntropyLoss()(pred, teacher)
        if self._loss_type == "dice":
            return DiceLoss()(pred, teacher)


class AuxiliarySegmentationLoss(nn.Module):
    def __init__(self, loss_type: str, auxiliary_loss_weight=0.4):
        super(AuxiliarySegmentationLoss, self).__init__()
        self._loss_func = SegmentationTypedLoss(loss_type)
        self._auxiliary_loss_func = SegmentationTypedLoss(loss_type)
        self._auxiliary_loss_weight = auxiliary_loss_weight

    def forward(self, pred: torch.Tensor, auxiliary_pred: torch.Tensor, teacher: torch.Tensor):
        return self._loss_func(pred, teacher) + self._auxiliary_loss_func(auxiliary_pred,
                                                                          teacher) * self._auxiliary_loss_weight


class AdaptiveCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor):
        """
        :param pred: (B * C * H * W) or (B * C)
        :param teacher: (B * C * H * W) or (B * H * W) or (B * C) or (B)
        """
        if pred.ndim == teacher.ndim:  # For One-Hot
            # TODO Label smoothingやりたいが今のところ無理っぽい
            return F.cross_entropy(pred, teacher.argmax(1), reduction="mean")
            # return nn.BCEWithLogitsLoss()(pred, teacher)
            # return F.binary_cross_entropy(nn.LogSoftmax(dim=1)(pred), teacher, reduction="mean")
        if pred.ndim == 4 and teacher.ndim == 3:
            return F.cross_entropy(pred, teacher, reduction="mean")
        if pred.ndim == 2 and teacher.ndim == 1:
            return F.cross_entropy(pred, teacher, reduction="mean")
        assert False, f"Invalid pred or teacher type,  {pred.shape} and {teacher.shape}"
