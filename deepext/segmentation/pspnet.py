import numpy as np
import torch
from torch import nn

from ..base import SegmentationModel
from ..utils import try_cuda
from ..layers import *
from ..utils import *


class PSPNet(SegmentationModel, nn.Module):
    def __init__(self, img_size, n_classes, in_channels=3, pooling_sizes=(6, 3, 2, 1), first_layer_channels=64, lr=1e-3,
                 loss_type="ce"):
        super(PSPNet, self).__init__()

        self._first_layer_channels = first_layer_channels
        self.in_channels, self.n_classes = in_channels, n_classes

        self._backbone: FeatureMapBackBone = FeatureMapBackBone(in_channels=in_channels,
                                                                first_layer_channels=first_layer_channels)

        backbone_out_channels = self._backbone.output_filter_num()

        self._pyramid_pooling = FeaturePyramidPooling(in_channels=backbone_out_channels,
                                                      compress_sizes=pooling_sizes)

        self._decoder = FeatureDecoder(in_channels=backbone_out_channels * 2, n_classes=n_classes, heigh=img_size[0],
                                       width=img_size[1], dropout_rate=0.1)
        self._auxiliary_decoder = FeatureDecoder(in_channels=backbone_out_channels, n_classes=n_classes,
                                                 heigh=img_size[0],
                                                 width=img_size[1], dropout_rate=0.1)

        self._optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
        self._loss_func = AuxiliarySegmentationLoss(loss_type=loss_type, auxiliary_loss_weight=0.4)

    def backbone_network(self, in_channels: int, first_layer_channels: int):
        return FeatureMapBackBone(in_channels=in_channels, first_layer_channels=first_layer_channels)

    def forward(self, x):
        """
        :param x: (batch size, channels, height, width)
        :return:  (batch size, class, height, width),  (batch size, class, height, width)
        """
        x = try_cuda(x)
        x = self._backbone(x)
        auxiliary_output = self._auxiliary_decoder(x)
        x = self._pyramid_pooling(x)
        output = self._decoder(x)
        return output, auxiliary_output

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class, height, width)
        """
        assert x.ndim == 4
        self.eval()
        x = try_cuda(x)
        return self.forward(x)[0].detach().cpu().numpy()

    def init_weights(self):
        init_weights_func(self._backbone)
        init_weights_func(self._pyramid_pooling)
        init_weights_func(self._auxiliary_decoder)
        init_weights_func(self._decoder)

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param train_x: (batch size, channels, height, width)
        :param teacher: (batch size, class, height, width)
        """
        self.train()
        train_x = try_cuda(train_x)
        teacher = try_cuda(teacher)
        self._optimizer.zero_grad()
        pred, auxiliary_pred = self(train_x)

        loss = self._loss_func(pred, auxiliary_pred, teacher)
        loss.backward()
        self._optimizer.step()
        # result = np.argmax(pred.cpu().detach().numpy(), axis=1).astype("uint32")
        # print(np.bincount(result.flatten(), minlength=20), loss.item())
        # print(segmentation_accuracy(pred.argmax(1), teacher.argmax(1)), loss.item())
        return loss.item()

    def save_weight(self, save_path: str):
        torch.save({
            'first_layer_channels': self._first_layer_channels,
            'n_channels': self.in_channels,
            'n_classes': self.n_classes,
            'model_state_dict': self.state_dict(),
            "optimizer": self.get_optimizer().state_dict(),
        }, save_path)

    def load_weight(self, weight_path: str):
        params = torch.load(weight_path)
        self._first_layer_channels = params['first_layer_channels']
        self.n_classes = params['n_classes']
        self.in_channels = params['n_classes']
        self.load_state_dict(params['model_state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])

    def get_optimizer(self):
        return self._optimizer

    def get_model_config(self):
        return {}


class ResPSPNet(PSPNet):
    def __init__(self, img_size, n_classes, in_channels=3, pooling_sizes=(6, 3, 2, 1), lr=1e-3, loss_type="ce"):
        super().__init__(img_size, n_classes, in_channels=in_channels, pooling_sizes=pooling_sizes,
                         first_layer_channels=32, lr=lr, loss_type=loss_type)

    def backbone_network(self, in_channels: int, first_layer_channels: int):
        return ResNetBackBone(in_channels=in_channels, n_block=3, pretrained=True, resnet_type="resnet50")
