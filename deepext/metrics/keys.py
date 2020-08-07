from enum import Enum


class MetricKey(Enum):
    KEY_ALL = "all"
    KEY_BACKGROUND = "background"
    KEY_AVERAGE = "average"
    KEY_AVERAGE_WITHOUT_BACKGROUND = "average without background"
    KEY_TOTAL = "total"
    KEY_RECALL = "recall"
    KEY_PRECISION = "precision"
    KEY_F_SCORE = "f_score"
