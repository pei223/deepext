from typing import Tuple
from deepext.base import BaseModel
from ..layers import *
from ..utils import *


class AttentionBranchNetwork(nn.Module, BaseModel):
    def __init__(self, n_classes: int, in_channels: int = 3, first_layer_channels=32, n_blocks=3, lr=1e-4):
        super().__init__()
        assert in_channels == 3
        self._n_classes = n_classes
        self.feature_extractor = nn.Sequential(
            ResidualBlock(in_channels=in_channels,
                          mid_channels=first_layer_channels,
                          out_channels=first_layer_channels * 2, n_blocks=2),
            ResidualBlock(in_channels=first_layer_channels * 2,
                          mid_channels=first_layer_channels,
                          out_channels=first_layer_channels * 4, n_blocks=2),
            ResidualBlock(in_channels=first_layer_channels * 4,
                          mid_channels=first_layer_channels * 2,
                          out_channels=first_layer_channels * 8, n_blocks=2),
        )
        feature_filter_num = first_layer_channels * 8
        self.attention_branch = AttentionClassifierBranch(in_channels=feature_filter_num, n_classes=n_classes,
                                                          n_blocks=n_blocks)

        self.perception_branch = nn.Sequential()
        for i in range(n_blocks - 1):
            if i == 0:
                self.perception_branch.add_module(f"block{i + 1}",
                                                  BottleNeck(in_channels=feature_filter_num,
                                                             mid_channels=feature_filter_num,
                                                             out_channels=feature_filter_num, stride=2))
                continue
            self.perception_branch.add_module(f"block{i + 1}",
                                              BottleNeckIdentity(in_channels=feature_filter_num,
                                                                 out_channels=feature_filter_num))
        self.perception_branch.add_module(f"block{n_blocks}",
                                          nn.Conv2d(kernel_size=1, padding=0, in_channels=feature_filter_num,
                                                    out_channels=n_classes))
        self.perception_branch.add_module("gap", GlobalAveragePooling())
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class), (batch size, class), heatmap (batch size, 1, height, width)
        """
        origin_feature = self.feature_extractor(x)
        attention_output, attention_map = self.attention_branch(origin_feature)
        # 特徴量・Attention mapのSkip connection
        perception_feature = origin_feature * attention_map
        perception_feature = perception_feature + origin_feature
        perception_output = self.perception_branch(perception_feature)
        perception_output = perception_output.view(perception_output.shape[0], -1)
        perception_output = F.softmax(perception_output, dim=1)
        return perception_output, attention_output, attention_map

    def train_batch(self, inputs: torch.Tensor, teachers: torch.Tensor) -> float:
        """
        :param inputs: (batch size, channels, height, width)
        :param teachers: (batch size, class)
        """
        self.train()
        inputs, teachers = try_cuda(inputs).float(), try_cuda(teachers).long()
        self._optimizer.zero_grad()
        pred = self(inputs)
        loss = self._calc_loss(pred, teachers)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def _calc_loss(self, output, teacher):
        teacher = teacher.long()
        perception_pred, attention_pred, attention_map = output
        return F.cross_entropy(perception_pred, teacher, reduction="mean") + F.cross_entropy(attention_pred, teacher,
                                                                                             reduction="mean")

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = try_cuda(x).float()
            return self(x)[0].cpu().numpy()

    def save_weight(self, save_path: str):
        dict_to_save = {
            'num_class': self._n_classes,
            'state_dict': self.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params['num_class']))
        self._n_classes = params['num_class']
        self.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        return self

    def get_model_config(self):
        return {
            'model_name': 'AttentionBranchNetwork',
            'num_classes': self._n_classes,
            'optimizer': self._optimizer.__class__.__name__
        }

    def get_optimizer(self):
        return self._optimizer


class ResNetAttentionBranchNetwork(AttentionBranchNetwork):
    def __init__(self, n_classes: int, in_channels: int = 3, pretrained=True, resnet_type="resnet50", n_blocks=3,
                 lr=1e-4, first_layer_channels=32):
        super().__init__(n_classes=n_classes, in_channels=in_channels, lr=lr)
        assert in_channels == 3
        self.feature_extractor: ResNetBackBone = ResNetBackBone(resnet_type=resnet_type, pretrained=pretrained)
        feature_filter_num = self.feature_extractor.output_filter_num()
        self.attention_branch = AttentionClassifierBranch(in_channels=feature_filter_num, n_classes=n_classes,
                                                          n_blocks=n_blocks)

        self.perception_branch = nn.Sequential()
        for i in range(n_blocks - 1):
            if i == 0:
                self.perception_branch.add_module(f"block{i + 1}",
                                                  BottleNeck(in_channels=feature_filter_num,
                                                             mid_channels=feature_filter_num,
                                                             out_channels=feature_filter_num, stride=2))
                continue
            self.perception_branch.add_module(f"block{i + 1}",
                                              BottleNeckIdentity(in_channels=feature_filter_num,
                                                                 out_channels=feature_filter_num))
        self.perception_branch.add_module(f"block{n_blocks}",
                                          nn.Conv2d(kernel_size=1, padding=0, in_channels=feature_filter_num,
                                                    out_channels=n_classes))
        self.perception_branch.add_module("gap", GlobalAveragePooling())
