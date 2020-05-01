import torch
from torch import nn
from ..utils.tensor_util import try_cuda
from ..layers import *
from ..base import SegmentationModel


class UNet(SegmentationModel, nn.Module):
    def __init__(self, n_input_channels, n_output_channels, first_layer_channels: int = 64, lr=1e-3, loss_type="ce"):
        super(UNet, self).__init__()

        self._first_layer_channels = first_layer_channels
        self.n_channels, self.n_classes = n_input_channels, n_output_channels
        self._encoder_layer1 = self.down_sampling_layer(n_input_channels, first_layer_channels)
        self._encoder_layer2 = self.down_sampling_layer(first_layer_channels, first_layer_channels * 2)
        self._encoder_layer3 = self.down_sampling_layer(first_layer_channels * 2, first_layer_channels * 4)
        self._encoder_layer4 = self.down_sampling_layer(first_layer_channels * 4, first_layer_channels * 8)
        self._encoder_layer5 = self.down_sampling_layer(first_layer_channels * 8, first_layer_channels * 16)

        self._decoder_layer1 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 8)
        self._decoder_layer2 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 4)
        self._decoder_layer3 = self.up_sampling_layer(first_layer_channels * 8, first_layer_channels * 2)
        self._decoder_layer4 = self.up_sampling_layer(first_layer_channels * 4, first_layer_channels)
        self._decoder_layer5 = self.up_sampling_layer(first_layer_channels * 2, n_output_channels, is_output_layer=True)

        self.apply(init_weights_func)

        self._optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
        self._loss_func = SegmentationTypedLoss(loss_type=loss_type)

    def forward(self, x):
        x = try_cuda(x)
        enc1 = self._encoder_layer1(x)
        enc2 = self._encoder_layer2(enc1)
        enc3 = self._encoder_layer3(enc2)
        enc4 = self._encoder_layer4(enc3)
        encoded_feature = self._encoder_layer5(enc4)

        x = self._decoder_layer1(encoded_feature)
        x = torch.cat([x, enc4], dim=1)
        x = self._decoder_layer2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self._decoder_layer3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self._decoder_layer4(x)
        x = torch.cat([x, enc1], dim=1)
        output = self._decoder_layer5(x)
        return output

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        :param x:
        :return: PIL形式のnumpy配列
        """
        assert x.ndim == 4
        self.eval()
        x = try_cuda(x)
        return self.forward(x).detach().cpu().numpy()

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return DownBlock(n_input_channels, n_out_channels)

    def up_sampling_layer(self, n_input_channels: int, n_out_channels: int, is_output_layer=False):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return UpBlock(n_input_channels, n_out_channels, is_output_layer)

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        self.train()
        train_x = try_cuda(train_x)
        teacher = try_cuda(teacher)
        self._optimizer.zero_grad()
        pred = self(train_x)
        loss = self._loss_func(pred, teacher)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def get_optimizer(self):
        return self._optimizer

    def save_weight(self, save_path: str):
        torch.save({
            'first_layer_channels': self._first_layer_channels,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'model_state_dict': self.state_dict(),
            "optimizer": self.get_optimizer().state_dict(),
        }, save_path)

    def load_weight(self, weight_path: str):
        params = torch.load(weight_path)
        self._first_layer_channels = params['first_layer_channels']
        self.n_classes = params['n_classes']
        self.n_channels = params['n_classes']
        self.load_state_dict(params['model_state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])

    def get_model_config(self):
        return {}


def init_weights_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)


class ResUNet(UNet):
    def __init__(self, n_input_channels, n_output_channels, lr=1e-3, loss_type="ce"):
        super().__init__(n_input_channels, n_output_channels, lr=lr, loss_type=loss_type)

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        return BottleNeck(n_input_channels, mid_channels=n_input_channels, out_channels=n_out_channels, stride=2)
