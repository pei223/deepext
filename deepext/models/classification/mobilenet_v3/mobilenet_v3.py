import torch.optim as optim
import torch
import numpy as np
from torch import nn
from ...base import ClassificationModel
from .mobilenetv3_lib.model import MobileNetV3 as MobileNetV3lib
from ....utils.tensor_util import try_cuda

__all__ = ['MobileNetV3']


class MobileNetV3(ClassificationModel):
    def __init__(self, num_classes, lr=1e-4, mode='small', pretrained=True):
        super().__init__()
        self._num_classes = num_classes
        self._mode = mode
        self._model = MobileNetV3lib(num_classes=num_classes, mode=mode)
        # state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._model.cuda()
            self._criterion.cuda()

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param train_x: (batch size, channel, height, width)
        :param teacher: (batch size, )
        """
        self._model.train()
        train_x = try_cuda(train_x).float()
        teacher = try_cuda(teacher).long()

        # compute output
        output = self._model(train_x)
        loss = self._criterion(output, teacher)
        # compute gradient and do SGD step
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def predict(self, inputs) -> np.ndarray:
        """
        :param inputs: (batch size, channel, height, width)
        :return: (batch size, class)
        """
        self._model.eval()
        with torch.no_grad():
            inputs = try_cuda(inputs).float()
            output = self._model(inputs)
            pred_ids = output.cpu().numpy()
        return pred_ids

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params['num_class']))
        self._num_classes = params['num_class']
        self._model.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        return self

    def get_model_config(self):
        config = {'model_name': 'MobileNetV3', 'num_classes': self._num_classes,
                  'optimizer': self._optimizer.__class__.__name__, 'mode': self._mode}
        return config

    def get_optimizer(self):
        return self._optimizer

    def get_model(self) -> nn.Module:
        return self._model
