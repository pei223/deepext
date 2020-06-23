from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ...base import SegmentationModel

from ...layers import ResNetMultiScaleBackBone, SegmentationTypedLoss, FocalLoss
from .modules import SegmentationShelf, OutLayer
from ...utils import try_cuda
from ...layers import Conv2DBatchNorm


class CustomShelfNet(SegmentationModel):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], in_channels=3, lr=1e-3, loss_type="ce",
                 backbone="resnet18"):
        super().__init__()
        self._n_classes = n_classes
        self._model: nn.Module = try_cuda(ShelfNetModel(n_classes=n_classes, out_size=out_size, in_channels=in_channels,
                                                        backbone=backbone))
        self._optimizer = torch.optim.Adam(lr=lr, params=self._model.parameters())
        # self._loss_func = FocalLoss()
        self._loss_func = SegmentationTypedLoss(loss_type=loss_type)

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
    def __init__(self, n_classes: int, out_size: Tuple[int, int], in_channels=3, backbone="resnet18"):
        super().__init__()

        if backbone in ["resnet18", "resnet34"]:
            backbone_out_channel_ls = [64, 128, 256, 512]
            mid_channel_ls = [64, 128, 256, 512]
        elif backbone == "resnet50":
            backbone_out_channel_ls = [256, 512, 1024, 2048]
            mid_channel_ls = [128, 256, 512, 1024]
        elif backbone == "resnet101":
            backbone_out_channel_ls = [512, 1024, 2048, 4096]
            mid_channel_ls = [256, 512, 1024, 2048]
        else:
            assert False, "Invalid backbone type."

        self._reducer_a = Conv2DBatchNorm(kernel_size=1, in_channels=backbone_out_channel_ls[0],
                                          out_channels=mid_channel_ls[0], padding=0)
        self._reducer_b = Conv2DBatchNorm(kernel_size=1, in_channels=backbone_out_channel_ls[1],
                                          out_channels=mid_channel_ls[1], padding=0)
        self._reducer_c = Conv2DBatchNorm(kernel_size=1, in_channels=backbone_out_channel_ls[2],
                                          out_channels=mid_channel_ls[2], padding=0)
        self._reducer_d = Conv2DBatchNorm(kernel_size=1, in_channels=backbone_out_channel_ls[3],
                                          out_channels=mid_channel_ls[3], padding=0)

        self._multi_scale_backbone = ResNetMultiScaleBackBone(resnet_type=backbone)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)
        self.conv_out = OutLayer(in_channels=mid_channel_ls[0], mid_channels=mid_channel_ls[0], n_classes=n_classes,
                                 out_size=out_size)
        self.conv_out_b = OutLayer(in_channels=mid_channel_ls[1], mid_channels=mid_channel_ls[0], n_classes=n_classes,
                                   out_size=out_size)
        self.conv_out_c = OutLayer(in_channels=mid_channel_ls[2], mid_channels=mid_channel_ls[1], n_classes=n_classes,
                                   out_size=out_size)

    def forward(self, x, aux=True):
        x_a, x_b, x_c, x_d = self._multi_scale_backbone(x)
        x_a = self._reducer_a(x_a)
        x_b = self._reducer_b(x_b)
        x_c = self._reducer_c(x_c)
        x_d = self._reducer_d(x_d)

        out_a, out_b, out_c = self._segmentation_shelf([x_a, x_b, x_c, x_d])
        if not aux:
            return self.conv_out(out_a)
        return self.conv_out(out_a), self.conv_out_b(out_b), self.conv_out_c(out_c)
