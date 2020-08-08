import math
from pathlib import Path
from ...models.base import BaseModel


class LearningRateScheduler:
    def __init__(self, max_epoch: int, power=0.9):
        self._max_epoch = max_epoch
        self._power = power

    def __call__(self, epoch: int):
        return math.pow((1 - epoch / self._max_epoch), self._power)


class WarmUpLRScheduler:
    def __init__(self, warmup_lr=1e-2, lr=1e-4, warmup_epochs=10):
        """
        :param warmup_lr: Warm up中の学習率
        :param lr: Warm up終了後の学習率
        :param warmup_epochs: 何エポックまでをWarm up期間とするか
        """
        self._warmup_lr = warmup_lr
        self._warmup_epochs = warmup_epochs
        self._lr = lr

    def __call__(self, epoch: int):
        return self._warmup_lr if epoch < self._warmup_epochs else self._lr


class ModelCheckout:
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
