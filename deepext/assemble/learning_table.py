from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import List, Callable


@dataclass
class LearningTable:
    epochs: int
    data_loader: DataLoader  # バギングを考慮
    callbacks: List[Callable[[int, ], None]] or None
