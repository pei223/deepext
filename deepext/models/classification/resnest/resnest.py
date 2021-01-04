import torch.nn as nn
import torch


class ResNeStModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self._net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

    def forward(self, x):
        return self._net(x)
