import math


class LearningRateScheduler:
    def __init__(self, base_lr: float, max_epoch: int, power=0.9):
        self._max_epoch = max_epoch
        self._power = power
        self._base_lr = base_lr

    def __call__(self, epoch: int):
        return (1 - max(epoch - 1, 1) / self._max_epoch) ** self._power * self._base_lr


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


class CosineDecayScheduler:
    def __init__(self, max_epochs: int, max_lr=0.16, warmup_epochs=5, min_lr=0):
        self._max_epochs = max_epochs
        self._max_lr = max_lr
        self._min_lr = min_lr
        self._warmup_epochs = warmup_epochs

    def __call__(self, epoch: int):
        epoch = max(epoch, 1)
        if epoch <= self._warmup_epochs:
            return self._max_lr * epoch / self._warmup_epochs
        epoch -= 1
        rad = math.pi * epoch / self._max_epochs
        weight = (math.cos(rad) + 1.) / 2

        return (self._max_lr - self._min_lr) * weight + self._min_lr
