from typing import List
from ..models.base import BaseModel
import torch
from collections import OrderedDict
from pathlib import Path
import pickle


class AssembleModel:
    MODEL_CONFIG_FILE = "config.pkl"

    def __init__(self, models: List[BaseModel]):
        self._models = models
        self._model_weights = [1 / len(self._models) for _ in range(len(self._models))]

    def predict(self, inputs: torch.Tensor):
        total_result = None
        for i, model in enumerate(self._models):
            if total_result is None:
                total_result = model.predict(inputs)
                continue
            total_result += model.predict(inputs) * self._model_weights[i]
        return total_result

    def save_weight(self, save_dir_path: str):
        save_dir = Path(save_dir_path)
        for i, model in enumerate(self._models):
            save_filepath = save_dir.joinpath(f"model_{i}.pth")
            model.save_weight(str(save_filepath))
        with open(save_dir.joinpath(self.MODEL_CONFIG_FILE), "w") as file:
            pickle.dump({"model_weights": self._model_weights}, file)

    def load_weight(self, weight_dir_path: str):
        load_dir = Path(weight_dir_path)
        for i, model in enumerate(self._models):
            load_filepath = load_dir.joinpath(f"model_{i}.pth")
            model.load_weight(str(load_filepath))
        with open(load_dir.joinpath(self.MODEL_CONFIG_FILE), "r") as file:
            params = pickle.load(file)
            self._model_weights = params["model_weights"]

    def get_model_config(self):
        config = OrderedDict()
        for i, model in enumerate(self._models):
            config[f"model_{i}"] = model.get_model_config()
        return config

    @property
    def model_list(self) -> List[BaseModel]:
        return self._models
