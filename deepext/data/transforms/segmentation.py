from ..transforms import ImageToOneHot


class SegmentationAlbumentationsToModel:
    def __init__(self, class_num: int, require_onehot=False):
        self.require_onehot = require_onehot
        self.class_num = class_num

    def __call__(self, data):
        image = data["image"]
        mask = data["mask"]
        if self.require_onehot and mask.ndim == 2:
            mask = ImageToOneHot(self.class_num)
        return image, mask
