from typing import Dict
from matplotlib import pyplot as plt


def visualize_category_hist(title: str, val_dict: Dict[str, int], out_filepath: str):
    label = list(val_dict.keys())
    y = list(val_dict.values())
    x = [_ for _ in range(len(y))]
    plt.bar(x, y, tick_label=label, align="center")
    plt.title(title)
    plt.savefig(out_filepath)
    plt.close()
