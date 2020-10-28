from ..base import AttentionClassificationModel
from ...layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from ...layers.subnetwork import *
from ...layers.block import *
from ...layers.basic import *
from ...utils import *
import torch

__all__ = ['AttentionBranchNetwork']


class AttentionBranchNetwork(AttentionClassificationModel):
    def __init__(self, n_classes: int, pretrained=True,
                 backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3, lr=1e-4):
        super().__init__()
        self._backbone = backbone
        self.model = try_cuda(
            ABNModel(n_classes=n_classes, pretrained=pretrained, backbone=backbone, n_blocks=n_blocks))
        self._n_classes = n_classes
        self._n_blocks = n_blocks
        self._optimizer = torch.optim.Adam(lr=lr, params=self.model.parameters())

    def train_batch(self, inputs: torch.Tensor, teachers: torch.Tensor) -> float:
        """
        :param inputs: (batch size, channels, height, width)
        :param teachers: (batch size, class)
        """
        self.model.train()
        inputs, teachers = try_cuda(inputs).float(), try_cuda(teachers).long()
        self._optimizer.zero_grad()
        pred = self.model(inputs)
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
        self.model.eval()
        with torch.no_grad():
            x = try_cuda(x).float()
            return self.model(x)[0].cpu().numpy()

    def predict_label_and_heatmap_impl(self, x) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x = try_cuda(x).float()
            pred, _, heatmap = self.model(x)
            pred, heatmap = pred.cpu().numpy(), heatmap[:, 0].cpu().numpy()
            heatmap = self._normalize_heatmap(heatmap)
            return pred, heatmap

    def _normalize_heatmap(self, heatmap: np.ndarray):
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        return (heatmap - min_val) / (max_val - min_val)

    def save_weight(self, save_path: str):
        dict_to_save = {
            'num_class': self._n_classes,
            'state_dict': self.model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params['num_class']))
        self._n_classes = params['num_class']
        self.model.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        return self

    def get_model_config(self):
        return {
            'model_name': 'AttentionBranchNetwork',
            'backbone': self._backbone.value,
            'num_classes': self._n_classes,
            'optimizer': self._optimizer.__class__.__name__
        }

    def get_optimizer(self):
        return self._optimizer


class ABNModel(nn.Module):
    def __init__(self, n_classes: int, pretrained=True,
                 backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3):
        super().__init__()
        self.feature_extractor = create_backbone(backbone_key=backbone, pretrained=pretrained)
        feature_channel_num = BACKBONE_CHANNEL_COUNT_DICT[backbone][-1]
        self.attention_branch = AttentionClassifierBranch(in_channels=feature_channel_num, n_classes=n_classes,
                                                          n_blocks=n_blocks)
        self.perception_branch = ClassifierHead(in_channels=feature_channel_num, n_blocks=n_blocks,
                                                n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class), (batch size, class), heatmap (batch size, 1, height, width)
        """
        origin_feature = self.feature_extractor(x)[-1]
        attention_output, attention_map = self.attention_branch(origin_feature)
        # 特徴量・Attention mapのSkip connection
        perception_feature = (origin_feature * attention_map) + origin_feature
        perception_output = self.perception_branch(perception_feature)
        return perception_output, attention_output, attention_map
