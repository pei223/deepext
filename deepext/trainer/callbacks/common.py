from pathlib import Path
from ...models.base import BaseModel
from abc import ABCMeta, abstractmethod


class ModelCallback(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, epoch: int):
        pass


class ModelCheckout(ModelCallback):
    def __init__(self, model: BaseModel, our_dir: str, per_epoch: int = 5):
        self._per_epoch = per_epoch
        self._out_dir = Path(our_dir)
        self._model = model
        if not self._out_dir.exists():
            self._out_dir.mkdir()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        self._model.save_weight(str(self._out_dir.joinpath(f"{self._model.__class__.__name__}_ep{epoch + 1}.pth")))
