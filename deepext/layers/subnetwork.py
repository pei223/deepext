import numpy as np

from .block import *
from .basic import *
from torch.nn import functional as F
import torchvision
from ..utils import *


class FeatureMapBackBone(nn.Module):
    def __init__(self, in_channels=3, first_layer_channels=32):
        super().__init__()
        n_block_list = [3, 4, 6, 3]

        self._feature_map_layer1 = Conv2DBatchNormRelu(in_channels, first_layer_channels, stride=2)
        self._feature_map_layer2 = Conv2DBatchNormRelu(first_layer_channels, first_layer_channels)
        self._feature_map_layer3 = Conv2DBatchNormRelu(first_layer_channels, first_layer_channels * 2)
        self._pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._residual_layer1 = ResidualBlock(in_channels=first_layer_channels * 2,
                                              mid_channels=first_layer_channels,
                                              stride=1,
                                              out_channels=first_layer_channels * 4, n_blocks=n_block_list[0])

        self._residual_layer2 = ResidualBlock(in_channels=first_layer_channels * 4,
                                              mid_channels=first_layer_channels * 2,
                                              out_channels=first_layer_channels * 8,
                                              stride=2,
                                              n_blocks=n_block_list[1])

        self._dilated_layer1 = ResidualBlock(in_channels=first_layer_channels * 8,
                                             mid_channels=first_layer_channels * 4,
                                             out_channels=first_layer_channels * 16, dilation=2,
                                             n_blocks=n_block_list[2])

        self._dilated_layer2 = ResidualBlock(in_channels=first_layer_channels * 16,
                                             mid_channels=first_layer_channels * 8,
                                             out_channels=first_layer_channels * 32, dilation=4,
                                             n_blocks=n_block_list[3])

        self._out_channels = first_layer_channels * 32

    def forward(self, x):
        x = self._feature_map_layer1(x)
        x = self._feature_map_layer2(x)
        x = self._feature_map_layer3(x)
        x = self._pool(x)
        x = self._residual_layer1(x)
        x = self._residual_layer2(x)
        x = self._dilated_layer1(x)
        x = self._dilated_layer2(x)
        return x

    def output_filter_num(self):
        return self._out_channels


class VGGFeatureBackBone(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, n_vgg_layer=18):
        super().__init__()
        assert in_channels == 3, "VGGのフィルタサイズに合わせるため"
        self._model = nn.Sequential()
        self._n_vgg_layer = n_vgg_layer
        vgg_feature = torchvision.models.vgg19(pretrained=pretrained).features
        self._model.add_module(
            "vgg_feature",
            vgg_feature[:n_vgg_layer]
        )

    def forward(self, x):
        return self._model(x)

    def output_filter_num(self):
        if self._n_vgg_layer < 5:
            return 64
        elif self._n_vgg_layer < 10:
            return 128
        elif self._n_vgg_layer < 19:
            return 256
        else:
            return 512


class ResNetBackBone(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, n_block: int = 3, resnet_type="resnet50"):
        super().__init__()
        assert in_channels == 3, "Resnetのフィルタサイズに合わせるため"
        assert 1 <= n_block <= 4
        assert resnet_type in ["resnet50", "resnet101"]
        resnet_model = torchvision.models.resnet50(
            pretrained=pretrained) if resnet_type == "resnet50" else torchvision.models.resnet101(pretrained=pretrained)
        self._n_block = n_block

        self._model = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
        )
        self._model.add_module("resnet_block1", resnet_model.layer1)
        if n_block < 2:
            return
        self._model.add_module("resnet_block2", resnet_model.layer2)
        if n_block < 3:
            return
        self._model.add_module("resnet_block3", resnet_model.layer3)
        if n_block < 4:
            return
        self._model.add_module("resnet_block4", resnet_model.layer4)

    def forward(self, x):
        return self._model(x)

    def output_filter_num(self):
        if self._n_block == 1:
            return 256
        elif self._n_block == 2:
            return 512
        elif self._n_block == 3:
            return 1024
        elif self._n_block == 4:
            return 2048
        assert False


class FeatureDecoder(nn.Module):
    def __init__(self, in_channels: int, heigh: int, width: int, n_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self._size = (heigh, width)
        self._upsampling1 = UpSamplingBlock(in_channels=in_channels, out_channels=int(in_channels / 4))
        self._upsampling2 = UpSamplingBlock(in_channels=int(in_channels / 4), out_channels=int(in_channels / 8))
        self._upsampling3 = UpSamplingBlock(in_channels=int(in_channels / 8), out_channels=int(in_channels / 16))

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.classification = Conv2DBatchNormRelu(in_channels=int(in_channels / 16), out_channels=n_classes,
                                                  kernel_size=1, padding=0)
        self._upsampling = nn.UpsamplingBilinear2d(size=self._size)
        self._output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._upsampling1(x)
        x = self._upsampling2(x)
        x = self._upsampling3(x)
        x = self.dropout1(x)
        x = self.classification(x)
        output = self._upsampling(x)
        output = self._output(output)
        return output


class AttentionClassifierBranch(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_blocks=3):
        super().__init__()
        assert n_blocks > 0
        self.conv_layers = nn.Sequential()
        for i in range(n_blocks):
            self.conv_layers.add_module(f"block{i + 1}",
                                        BottleNeckIdentity(in_channels=in_channels, out_channels=in_channels))

        self.conv_layers.add_module("bn", nn.BatchNorm2d(in_channels))
        self.conv_layers.add_module("1x1conv",
                                    nn.Conv2d(in_channels=in_channels, out_channels=n_classes, kernel_size=1,
                                              padding=0))
        self.conv_layers.add_module("relu", nn.ReLU())

        self.attention_mapper = nn.Sequential(
            nn.Conv2d(in_channels=n_classes, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=1, padding=0),
            GlobalAveragePooling(),
        )

    def forward(self, feature):
        feature = self.conv_layers(feature)
        attention_map = self.attention_mapper(feature)
        classification_label = self.classifier(feature).view(feature.shape[0], -1)
        classification_label = F.softmax(input=classification_label, dim=1)
        return classification_label, attention_map


class ResNetMultiScaleBackBone(nn.Module):
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        assert resnet_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

    def forward(self, x):
        """
        :param x: (Batch size, 3, height, width)
        :return: (Batch size, 64, height/2, width/2), (Batch size, 128, height/4, width/4),
        (Batch size, 256, height/8, width/8), (Batch size, 512, height/16, width/16)
        """
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)
        out2 = self.resnet_model.layer2(out1)
        out3 = self.resnet_model.layer3(out2)
        out4 = self.resnet_model.layer4(out3)
        return out1, out2, out3, out4
