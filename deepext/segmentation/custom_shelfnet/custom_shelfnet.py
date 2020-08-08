from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ...base import SegmentationModel

from ...layers import ResNetMultiScaleBackBone, SegmentationTypedLoss, BACKBONE_CHANNEL_COUNT_DICT, BackBoneKey
from .modules import SegmentationShelf, OutLayer
from ...utils import try_cuda
from ...layers import Conv2DBatchNorm


class CustomShelfNet(SegmentationModel):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], lr=1e-3, loss_func: nn.Module = None,
                 backbone: BackBoneKey = BackBoneKey.RESNET_18):
        super().__init__()
        self._n_classes = n_classes
        self._model: nn.Module = try_cuda(ShelfNetModel(n_classes=n_classes, out_size=out_size, backbone=backbone))
        self._optimizer = torch.optim.Adam(lr=lr, params=self._model.parameters())
        self._loss_func = loss_func if loss_func else SegmentationTypedLoss(loss_type="ce")

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        self._model.train()
        train_x, teacher = try_cuda(train_x), try_cuda(teacher)
        self._optimizer.zero_grad()
        pred, pred_b, pred_c = self._model(train_x)[:3]
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
    def __init__(self, n_classes: int, out_size: Tuple[int, int], backbone: BackBoneKey = BackBoneKey.RESNET_18):
        super().__init__()

        backbone_out_channel_ls = BACKBONE_CHANNEL_COUNT_DICT.get(backbone)
        assert backbone_out_channel_ls is not None, "Invalid backbone type."
        if backbone in [BackBoneKey.RESNET_18, BackBoneKey.RESNET_34]:
            mid_channel_ls = [64, 128, 256]
        elif backbone == BackBoneKey.RESNET_50:
            mid_channel_ls = [128, 256, 512]
        elif backbone == BackBoneKey.RESNET_101:
            mid_channel_ls = [256, 512, 1024]
        else:
            assert False, "Invalid backbone type."

        reducers = []
        layer_diff = len(backbone_out_channel_ls) - len(mid_channel_ls)
        assert layer_diff >= 0, "Shelfのレイヤー数はBackBoneのレイヤー数以下の必要があります."
        for i in range(len(mid_channel_ls)):
            reducers.append(Conv2DBatchNorm(kernel_size=1, in_channels=backbone_out_channel_ls[layer_diff + i],
                                            out_channels=mid_channel_ls[i], padding=0))
        self._reducers = nn.ModuleList(reducers)

        self._multi_scale_backbone = ResNetMultiScaleBackBone(resnet_type=backbone)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)

        out_convs = []
        for i in range(len(mid_channel_ls)):
            out_convs.append(OutLayer(in_channels=mid_channel_ls[i], mid_channels=mid_channel_ls[0],
                                      n_classes=n_classes, out_size=out_size))
        self._out_convs = nn.ModuleList(out_convs)

    def forward(self, x, aux=True):
        x_list = self._multi_scale_backbone(x)[-len(self._reducers):]  # BackBoneの後ろからレイヤー数分取得する.
        reduced_x_list = []
        for i, x in enumerate(x_list):
            reduced_x_list.append(self._reducers[i](x))

        x_list = self._segmentation_shelf(reduced_x_list)
        if not aux:
            return self._out_convs[0](x_list[0])

        outs = []
        for i in range(len(self._out_convs)):
            outs.append(self._out_convs[i](x_list[i]))
        return outs
