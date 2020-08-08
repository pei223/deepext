from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from ...layers import Conv2DBatchNormRelu, SharedWeightResidualBlock, ChannelWiseAttentionBlock, Conv2DBatchNorm


class SegmentationShelf(nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()
        self._decoder = Decoder(in_channels_ls=in_channels_ls)
        self._encoder = Encoder(in_channels_ls=in_channels_ls)
        self._final_decoder = Decoder(in_channels_ls=in_channels_ls)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        dec_outputs = self._decoder(inputs)
        enc_outputs = self._encoder(dec_outputs)
        final_output = self._final_decoder(enc_outputs, contain_bottom=True)
        return final_output


class OutLayer(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int, out_size: Tuple[int, int]):
        super().__init__()
        self._out_size = out_size
        self.mid_conv = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                            padding=1)
        self.conv_out = nn.Conv2d(in_channels=mid_channels, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mid_conv(x)
        x = self.conv_out(x)
        return F.interpolate(x, size=self._out_size, mode='bilinear', align_corners=True)


class Decoder(nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()

        sblocks = list(map(lambda in_channel: SharedWeightResidualBlock(in_channels=in_channel), in_channels_ls[::-1]))
        self._sblocks = nn.ModuleList(sblocks)

        attention_connections, dense_connections = [], []
        for i in range(1, len(in_channels_ls)):
            attention = ChannelWiseAttentionBlock(in_channels=in_channels_ls[i], out_channels=in_channels_ls[i - 1])
            dense = Conv2DBatchNorm(in_channels=in_channels_ls[i - 1], out_channels=in_channels_ls[i - 1],
                                    kernel_size=3, stride=1, padding=1)
            attention_connections.append(attention)
            dense_connections.append(dense)

        self._attention_connections = nn.ModuleList(attention_connections[::-1])
        self._dense_connections = nn.ModuleList(dense_connections[::-1])

    def forward(self, inputs: List[torch.Tensor], contain_bottom=False):
        # inputsは高解像度から低解像度の順
        outs = []
        if contain_bottom:
            outs.append(inputs[-1])
        # 低解像度から高解像度へアップサンプリング
        out = self._sblocks[0](inputs[-1], None)
        for i in range(len(self._attention_connections)):
            out = self._attention_connections[i](out)
            out = self._upsampling(out)
            out = self._dense_connections[i](out)
            # SBlockは1つ多く進める
            input_ = inputs[-(i + 2)]  # 高解像度から低解像度順のため逆のインデックスになる
            out = self._sblocks[i + 1](input_, out)
            outs.insert(0, out)  # 高解像度から低解像度になるよう追加
        return outs

    def _upsampling(self, x):
        return F.interpolate(x, (x.size(2) * 2, x.size(3) * 2), mode="nearest")


class Encoder(nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()

        sblocks = list(map(lambda in_channel: SharedWeightResidualBlock(in_channels=in_channel), in_channels_ls[:-1]))
        self._sblocks = nn.ModuleList(sblocks)

        down_convs = []
        for i in range(1, len(in_channels_ls)):
            down_convs.append(nn.Conv2d(kernel_size=3, stride=2, in_channels=in_channels_ls[i - 1],
                                        out_channels=in_channels_ls[i], padding=1))
        self._down_convs = nn.ModuleList(down_convs)

    def forward(self, inputs: List[torch.Tensor]):
        out = None
        outs = []
        for i in range(len(self._sblocks)):
            out = self._sblocks[i](inputs[i], out)
            outs.append(out)
            out = self._down_convs[i](out)
        outs.append(out)
        return outs
