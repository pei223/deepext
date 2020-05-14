from torch.utils.data import DataLoader
from typing import List, Callable


class LearningTable:
    def __init__(self, epochs: int, data_loader: DataLoader, callbacks: List[Callable[[int, ], None]] or None):
        self.epochs = epochs
        self.data_loader = data_loader  # バギングを考慮
        self.callbacks = callbacks
