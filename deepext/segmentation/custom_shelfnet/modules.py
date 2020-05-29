from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from ...layers import Conv2DBatchNormRelu, SharedWeightResidualBlock, ChannelWiseAttentionBlock


class SegmentationShelf(nn.Module):
    def __init__(self, in_channels_ls=None):
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
        self._final_decoder = Decoder()

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
    def __init__(self, in_channels_ls=None):
        if in_channels_ls is None:
            in_channels_ls = [64, 128, 256, 512]
        assert len(in_channels_ls) == 4
        super().__init__()

        self._sblock_d = SharedWeightResidualBlock(in_channels=in_channels_ls[3])
        self._sblock_c = SharedWeightResidualBlock(in_channels=in_channels_ls[2])
        self._sblock_b = SharedWeightResidualBlock(in_channels=in_channels_ls[1])
        self._sblock_a = SharedWeightResidualBlock(in_channels=in_channels_ls[0])

        self._attention_d2c = ChannelWiseAttentionBlock(in_channels=in_channels_ls[3], out_channels=in_channels_ls[2])
        self._dense_d2c = Conv2DBatchNormRelu(in_channels=in_channels_ls[2], out_channels=in_channels_ls[2],
                                              kernel_size=3, stride=1, padding=1)
        self._attention_c2b = ChannelWiseAttentionBlock(in_channels=in_channels_ls[2], out_channels=in_channels_ls[1])
        self._dense_c2b = Conv2DBatchNormRelu(in_channels=in_channels_ls[1], out_channels=in_channels_ls[1],
                                              kernel_size=3, stride=1, padding=1)
        self._attention_b2a = ChannelWiseAttentionBlock(in_channels=in_channels_ls[1], out_channels=in_channels_ls[0])
        self._dense_b2a = Conv2DBatchNormRelu(in_channels=in_channels_ls[0], out_channels=in_channels_ls[0],
                                              kernel_size=3, stride=1, padding=1)

        # self._upconv_d2c = nn.ConvTranspose2d(in_channels=in_channels_ls[3], out_channels=in_channels_ls[2],
        #                                       kernel_size=2, stride=2)
        # self._upconv_c2b = nn.ConvTranspose2d(in_channels=in_channels_ls[2], out_channels=in_channels_ls[1],
        #                                       kernel_size=2, stride=2)
        # self._upconv_b2a = nn.ConvTranspose2d(in_channels=in_channels_ls[1], out_channels=in_channels_ls[0],
        #                                       kernel_size=2, stride=2)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        assert len(inputs) == 4
        input_a, input_b, input_c, input_d = inputs
        # out = self._sblock_d(input_d, None)
        # out = self._upconv_d2c(out)
        # out_c = self._sblock_c(input_c, out)
        # out = self._upconv_c2b(out_c)
        # out_b = self._sblock_b(input_b, out)
        # out = self._upconv_b2a(out_b)
        # out_a = self._sblock_a(input_a, out)
        out = self._sblock_d(input_d, None)
        out = self._attention_d2c(out)
        out = self._upsampling(out)
        out = self._dense_d2c(out)

        out_c = self._sblock_c(input_c, out)
        out = self._attention_c2b(out)
        out = self._upsampling(out)
        out = self._dense_c2b(out)

        out_b = self._sblock_b(input_b, out)
        out = self._attention_b2a(out)
        out = self._upsampling(out)
        out = self._dense_b2a(out)

        out_a = self._sblock_a(input_a, out)
        return out_a, out_b, out_c

    def _upsampling(self, x):
        return F.interpolate(x, (x.size(2) * 2, x.size(3) * 2), mode="nearest")


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
