from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ...base import SegmentationModel

from ...layers import ResNetMultiScaleBackBone, SegmentationTypedLoss, FocalLoss
from .modules import SegmentationShelf, OutLayer
from ...utils import try_cuda


class CustomShelfNet(SegmentationModel):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], in_channels=3, lr=1e-3, loss_type="ce"):
        super().__init__()
        self._n_classes = n_classes
        self._model: nn.Module = ShelfNetModel(n_classes=n_classes, out_size=out_size, in_channels=in_channels)
        self._optimizer = torch.optim.Adam(lr=lr, params=self._model.parameters())
        self._loss_func = FocalLoss()

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        self._model.train()
        train_x, teacher = try_cuda(train_x), try_cuda(teacher)
        self._optimizer.zero_grad()
        pred, pred_b, pred_c = self._model(train_x)
        loss_a = self._loss_func(pred, teacher)
        loss_b = self._loss_func(pred_b, teacher)
        loss_c = self._loss_func(pred_c, teacher)
        loss = loss_a + loss_b + loss_c

        loss.backward()
        self._optimizer.step()
        return loss.item()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class, height, width)
        """
        assert x.ndim == 4
        self._model.eval()
        x = try_cuda(x)
        pred = self._model(x, aux=False)
        return pred.detach().cpu().numpy()

    def save_weight(self, save_path: str):
        torch.save({
            'n_classes': self._n_classes,
            'model_state_dict': self._model.state_dict(),
            "optimizer": self.get_optimizer().state_dict(),
        }, save_path)

    def load_weight(self, weight_path: str):
        params = torch.load(weight_path)
        self._n_classes = params['n_classes']
        self._model.load_state_dict(params['model_state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])

    def get_optimizer(self):
        return self._optimizer

    def get_model_config(self):
        return {}


class ShelfNetModel(nn.Module):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], in_channels=3, resnet_type="resnet18"):
        super().__init__()
        self._multi_scale_backbone = ResNetMultiScaleBackBone(resnet_type=resnet_type)
        self._segmentation_shelf = SegmentationShelf()
        self.conv_out = OutLayer(in_channels=64, mid_channels=64, n_classes=n_classes, out_size=out_size)
        self.conv_out_b = OutLayer(in_channels=128, mid_channels=64, n_classes=n_classes, out_size=out_size)
        self.conv_out_c = OutLayer(in_channels=256, mid_channels=64, n_classes=n_classes, out_size=out_size)

    def forward(self, x, aux=True):
        x = self._multi_scale_backbone(x)
        out_a, out_b, out_c = self._segmentation_shelf(x)
        if not aux:
            return self.conv_out(out_a)
        return self.conv_out(out_a), self.conv_out_b(out_b), self.conv_out_c(out_c)
