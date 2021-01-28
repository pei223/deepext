from typing import Tuple
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from ..base import ClassificationModel
from ...layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from ...layers.subnetwork import create_backbone, ClassifierHead
from ...utils import try_cuda


class CustomClassificationNetwork(ClassificationModel):
    def __init__(self, n_classes: int, pretrained=True, backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3,
                 lr=1e-4, multi_class=False):
        super().__init__()
        self._backbone = backbone
        self._model = try_cuda(
            CustomClassificationModel(n_classes=n_classes, pretrained=pretrained, backbone=backbone, n_blocks=n_blocks))
        self._n_classes = n_classes
        self._n_blocks = n_blocks
        self._optimizer = torch.optim.Adam(lr=lr, params=self._model.parameters())
        self._multi_class = multi_class

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param train_x: (batch size, channels, height, width)
        :param teacher: (batch size, class)
        """
        self._model.train()
        train_x = try_cuda(train_x).float()
        teacher = try_cuda(teacher).long()

        # compute output
        output = self._model(train_x)
        if not self._multi_class:
            output = F.softmax(output, dim=1)
            loss = F.cross_entropy(output, teacher, reduction="mean")
        else:
            output = F.sigmoid(output)
            loss = F.binary_cross_entropy(output, teacher, reduction="mean")
        # compute gradient and do SGD step
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def predict(self, x):
        self._model.eval()
        with torch.no_grad():
            x = try_cuda(x).float()
            return self._model(x).cpu().numpy()

    def save_weight(self, save_path: str):
        dict_to_save = {
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path: str):
        params = torch.load(weight_path)
        self._model.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        return self

    def get_model_config(self):
        return {
            "model_name": "CustomClassificationNetwork",
            'backbone': self._backbone.value,
            'num_classes': self._n_classes,
            'optimizer': self._optimizer.__class__.__name__
        }

    def get_optimizer(self):
        return self._optimizer

    def get_model(self) -> nn.Module:
        return self._model


class CustomClassificationModel(nn.Module):
    def __init__(self, n_classes: int, pretrained=True,
                 backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3):
        super().__init__()
        self.feature_extractor = create_backbone(backbone_key=backbone, pretrained=pretrained)
        feature_channel_num = BACKBONE_CHANNEL_COUNT_DICT[backbone][-1]
        self.perception_branch = ClassifierHead(in_channels=feature_channel_num, n_blocks=n_blocks,
                                                n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class), (batch size, class), heatmap (batch size, 1, height, width)
        """
        origin_feature = self.feature_extractor(x)[-1]
        return self.perception_branch(origin_feature)
