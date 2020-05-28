from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from ...layers import Conv2DBatchNormRelu, SharedWeightResidualBlock


class SegmentationShelf(nn.Module):
    def __init__(self, out_channels: int, out_size: Tuple[int, int], in_channels_ls=None):
        if in_channels_ls is None:
            in_channels_ls = [64, 128, 256, 512]
        assert len(in_channels_ls) == 4
        super().__init__()
        self._conv1x1_a = Conv2DBatchNormRelu(kernel_size=1, in_channels=in_channels_ls[0],
                                              out_channels=in_channels_ls[0], padding=0)
        self._conv1x1_b = Conv2DBatchNormRelu(kernel_size=1, in_channels=in_channels_ls[1],
                                              out_channels=in_channels_ls[1], padding=0)
        self._conv1x1_c = Conv2DBatchNormRelu(kernel_size=1, in_channels=in_channels_ls[2],
                                              out_channels=in_channels_ls[2], padding=0)
        self._conv1x1_d = Conv2DBatchNormRelu(kernel_size=1, in_channels=in_channels_ls[3],
                                              out_channels=in_channels_ls[3], padding=0)
        self._decoder = Decoder()
        self._encoder = Encoder()
        self._final_decoder = OutputDecoder(out_channels=out_channels, out_size=out_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        assert len(inputs) == 4
        input_a, input_b, input_c, input_d = inputs
        input_a = self._conv1x1_a(input_a)
        input_b = self._conv1x1_b(input_b)
        input_c = self._conv1x1_c(input_c)
        input_d = self._conv1x1_d(input_d)

        dec_outputs = self._decoder((input_a, input_b, input_c, input_d))
        enc_outputs = self._encoder(dec_outputs)
        final_output = self._final_decoder(enc_outputs)
        return final_output


class Decoder(nn.Module):
    def __init__(self, in_channels_ls=None):
        if in_channels_ls is None:
            in_channels_ls = [64, 128, 256, 512]
        assert len(in_channels_ls) == 4
        super().__init__()

        self._sblock_d = SharedWeightResidualBlock(in_channels=in_channels_ls[3])
        self._sblock_c = SharedWeightResidualBlock(in_channels=in_channels_ls[2])
        self._sblock_b = SharedWeightResidualBlock(in_channels=in_channels_ls[1])
        self._sblock_a = SharedWeightResidualBlock(in_channels=in_channels_ls[0])

        self._upconv_d2c = nn.ConvTranspose2d(in_channels=in_channels_ls[3], out_channels=in_channels_ls[2],
                                              kernel_size=2, stride=2)
        self._upconv_c2b = nn.ConvTranspose2d(in_channels=in_channels_ls[2], out_channels=in_channels_ls[1],
                                              kernel_size=2, stride=2)
        self._upconv_b2a = nn.ConvTranspose2d(in_channels=in_channels_ls[1], out_channels=in_channels_ls[0],
                                              kernel_size=2, stride=2)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        assert len(inputs) == 4
        input_a, input_b, input_c, input_d = inputs
        out = self._sblock_d(input_d, None)
        out = self._upconv_d2c(out)
        out_c = self._sblock_c(input_c, out)
        out = self._upconv_c2b(out_c)
        out_b = self._sblock_b(input_b, out)
        out = self._upconv_b2a(out_b)
        out_a = self._sblock_a(input_a, out)
        return out_a, out_b, out_c


class OutputDecoder(Decoder):
    def __init__(self, out_channels: int, out_size: Tuple[int, int], in_channels_ls=None):
        if in_channels_ls is None:
            in_channels_ls = [64, 128, 256, 512]
        super().__init__(in_channels_ls=in_channels_ls)
        self._out_size = out_size
        self._out_layer = nn.Conv2d(kernel_size=3, stride=1, in_channels=in_channels_ls[0], out_channels=out_channels,
                                    padding=1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        outputs = super(OutputDecoder, self).forward(inputs)
        out = self._out_layer(outputs[0])
        out = F.interpolate(out, size=self._out_size, mode='bilinear', align_corners=True)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels_ls=None):
        if in_channels_ls is None:
            in_channels_ls = [64, 128, 256, 512]
        assert len(in_channels_ls) == 4
        super().__init__()
        self._sblock_a = SharedWeightResidualBlock(in_channels=in_channels_ls[0])
        self._sblock_b = SharedWeightResidualBlock(in_channels=in_channels_ls[1])
        self._sblock_c = SharedWeightResidualBlock(in_channels=in_channels_ls[2])
        self._sblock_d = SharedWeightResidualBlock(in_channels=in_channels_ls[3])

        self._downconv_a2b = nn.Conv2d(kernel_size=3, stride=2, in_channels=in_channels_ls[0],
                                       out_channels=in_channels_ls[1], padding=1)
        self._downconv_b2c = nn.Conv2d(kernel_size=3, stride=2, in_channels=in_channels_ls[1],
                                       out_channels=in_channels_ls[2], padding=1)
        self._downconv_c2d = nn.Conv2d(kernel_size=3, stride=2, in_channels=in_channels_ls[2],
                                       out_channels=in_channels_ls[3], padding=1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        assert len(inputs) == 3
        input_a, input_b, input_c = inputs
        out_a = self._sblock_a(input_a, None)
        out = self._downconv_a2b(out_a)
        out_b = self._sblock_b(input_b, out)
        out = self._downconv_b2c(out_b)
        out_c = self._sblock_c(input_c, out)
        out_d = self._downconv_c2d(out_c)
        return out_a, out_b, out_c, out_d
