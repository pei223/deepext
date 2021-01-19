from abc import ABCMeta, abstractmethod
from warnings import warn

import torch
import numpy as np
import torch.nn as nn
import torch
from torch.utils import mobile_optimizer

from ...utils import try_cuda


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def train_batch(self, train_x: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        1バッチの学習を進めてlossを返却する.
        :param train_x: B * Channel * H * W
        :param teacher: B * Class * H * W
        :return:
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def save_weight(self, save_path: str):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def load_weight(self, weight_path: str):
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    def save_model_for_mobile(self, width: int, height: int, out_filepath: str, for_os="cpu"):
        torch_model = self.get_model()
        torch_model = torch_model.to("cpu")
        torch_model.eval()

        if for_os == "cpu":
            example = torch.rand(1, 3, width, height).to("cpu")
            traced_script_module = torch.jit.trace(torch_model, example)
            traced_script_module.save(out_filepath)
            return

        script_model = torch.jit.script(torch_model)
        if for_os == "android":
            warn("Vulkan ")
            mobile_optimizer.optimize_for_mobile(script_model, backend="Vulkan")
        elif for_os == "ios":
            mobile_optimizer.optimize_for_mobile(script_model, backend="metal")
        torch.jit.save(script_model, out_filepath)

        # scripted_model = torch.jit.script(torch_model)
        # opt_model = mobile_optimizer.optimize_for_mobile(scripted_model)
        # torch.jit.save(opt_model, out_filepath, _use_new_zipfile_serialization=False)
