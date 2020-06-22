import torch
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
        pred = F.normalize(pred - pred.min(), 1)
        # print("pred", pred.sum(1, ))

        intersection = (pred * teacher.float()).sum((-1, -2))
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[0], teacher.shape[1], -1)

        total_area = float(teacher.shape[-1])
        area_by_classes = teacher.long().sum(-1, )
        weight_by_classes = (total_area - area_by_classes.float()) / total_area
        # print("weight by class", weight_by_classes)

        pred_sum = pred.sum((-1,))
        teacher_sum = teacher.long().sum((-1,))
        # Batch * Classes
        dice_loss = 1. - ((2. * intersection + smooth) / (pred_sum + teacher_sum.float() + smooth))
        dice_loss[dice_loss != dice_loss] = 1.
        # print("dice loss", dice_loss)
        weighted_dice_loss = (dice_loss * weight_by_classes).mean(-1, )  # Batch,
        # print("weighted_dice", weighted_dice_loss)

        loss = weighted_dice_loss.mean(0, )
        # print(loss)
        return loss if not torch.isnan(loss) else 1.0


class SegmentationTypedLoss(nn.Module):
    def __init__(self, loss_type: str):
        super(SegmentationTypedLoss, self).__init__()
        assert loss_type.lower() in ["ce+dice", "dice", "ce", "weighted_dice"]
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
        if self._loss_type == "weighted_dice":
            return DiceLossWithAreaPenalty()(pred, teacher)


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weights: torch.Tensor = None):
        # TODO Class weight対応
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, teacher):
        logpt = -F.cross_entropy(pred, teacher.argmax(1))
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt
        loss = loss.view(1, -1)
        return loss.mean(dim=-1)
