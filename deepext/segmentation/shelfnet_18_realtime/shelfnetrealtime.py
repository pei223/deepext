import torch
from typing import Tuple
from ...base import SegmentationModel
from .shelfnet_lib.mainmodel import MainModel
from .shelfnet_lib.optimizer import Optimizer
from .shelfnet_lib.loss import OhemCELoss
from ...layers import *


class ShelfNetRealtime(SegmentationModel):
    def __init__(self, num_classes, size: Tuple[int, int], batch_size_per_gpu=4, lr=1e-2):
        super().__init__()
        self._num_classes = num_classes
        self._model = try_cuda(MainModel(num_classes))
        self._optimizer = self._init_shelfnet_optimizer(self._model, lr=lr)
        self._criterion = self._init_shelfnet_loss(batch_size_per_gpu, size)

    def get_optimizer(self):
        return self._optimizer

    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor):
        teachers = torch.argmax(teacher, dim=1)

        self._optimizer.zero_grad()
        out, out16, out32 = self._model(train_x)
        lossp = self._criterion[0](out, teachers)
        loss2 = self._criterion[1](out16, teachers)
        loss3 = self._criterion[2](out32, teachers)
        loss = lossp + loss2 + loss3
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def predict(self, inputs):
        self._model.eval()
        with torch.no_grad():
            output, _, _ = self._model(inputs)
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
        config = {}
        config['model_name'] = 'ShelfNet'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        return config

    @staticmethod
    def _init_shelfnet_optimizer(net, lr: float):
        momentum = 0.9
        weight_decay = 5e-4
        max_iter = 80000
        power = 0.9
        warmup_steps = 1000
        warmup_start_lr = 1e-5
        optim = Optimizer(
            model=net,
            lr0=lr,
            momentum=momentum,
            wd=weight_decay,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            max_iter=max_iter,
            power=power)
        return optim

    @staticmethod
    def _init_shelfnet_loss(batch_size_per_gpu, cropsize, ignore_idx=255, score_thres=0.7):
        n_min = batch_size_per_gpu * cropsize[0] * cropsize[1] // 16
        LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        return LossP, Loss2, Loss3
