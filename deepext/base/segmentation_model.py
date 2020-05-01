from typing import List
from PIL import Image
from torchvision.transforms import ToTensor

from .base_model import BaseModel
from ..utils.tensor_util import try_cuda


class SegmentationModel(BaseModel):
    def predict_one_image(self, img: Image.Image, color_palette: List[int] or None = None) -> Image.Image:
        x = ToTensor()(img)
        x = x.view(-1, x.shape[0], x.shape[1], x.shape[2])
        x = try_cuda(x)
        result = self.predict(x)[0].argmax(axis=0).astype('uint32')
        result_img = Image.fromarray(result).convert(mode="P")
        if color_palette:
            result_img.putpalette(color_palette)
        return result_img
