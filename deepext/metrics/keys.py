from enum import Enum


class MainMetricKey(Enum):
    KEY_RECALL = "recall"
    KEY_PRECISION = "precision"
    KEY_F_SCORE = "f_score"


class DetailMetricKey(Enum):
    KEY_ALL = "all"
    KEY_BACKGROUND = "background"
    KEY_AVERAGE = "average"
    KEY_AVERAGE_WITHOUT_BACKGROUND = "average without background"
    KEY_TOTAL = "total"
