import numpy as np


class DetectionAlbumentationsToModel:
    def __call__(self, data):
        image = data["image"]
        bboxes = data["bboxes"]
        labels = data["category_id"]
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        annotation = np.concatenate([bboxes, labels], axis=1)
        return image, annotation
