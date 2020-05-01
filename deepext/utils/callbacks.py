from typing import List, Iterable
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from .image_utils import *
from ..base import SegmentationModel, BaseModel
from ..classifier import AttentionBranchNetwork


class GenerateSegmentationImageCallback:
    def __init__(self, model: SegmentationModel, output_dir: str, per_epoch: int, dataset: Dataset, alpha=150,
                 color_palette_ls: List[int] = None):
        self._model: SegmentationModel = model
        self._output_dir = output_dir
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._alpha = alpha
        self._to_pil = ToPILImage()
        self._color_palette = color_palette_ls or default_color_palette()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, _ = self._dataset[random_image_index]
        img = self._to_pil(img_tensor)
        result_img = self._model.predict_one_image(img, self._color_palette)

        result_img = transparent_only_black(result_img, self._alpha)
        blend_result_and_source(np.array(img.convert('RGB')), np.array(result_img)).save(
            f"{self._output_dir}/result_image{epoch + 1}.png")


class LearningRateScheduler:
    def __init__(self, max_epoch: int):
        self._max_epoch = max_epoch

    def __call__(self, epoch: int):
        return math.pow((1 - epoch / self._max_epoch), 0.9)


class GenerateAttentionMapCallback:
    def __init__(self, output_dir: str, per_epoch: int, dataset: Dataset, model: AttentionBranchNetwork):
        self._out_dir, self._per_epoch, self._dataset = output_dir, per_epoch, dataset
        self._model = model
        self._to_pil = ToPILImage()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        self._model.eval()
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, label = self._dataset[random_image_index]
        img_tensor = img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]).cuda()
        perception_pred, attention_pred, attention_map = self._model(img_tensor)
        pred_label = perception_pred.argmax(-1).item()

        img: Image.Image = self._to_pil(img_tensor.detach().cpu()[0])
        img.save(f"{self._out_dir}/epoch{epoch + 1}_t{label}_p{pred_label}.png")

        plt.figure()
        sns.heatmap(attention_map.cpu().detach().numpy()[0][0])
        plt.savefig(f"{self._out_dir}/epoch{epoch + 1}_attention.png")
        plt.close('all')


class VisualizeObjectDetectionResult:
    def __init__(self, model: BaseModel, dataset: Dataset, out_dir: str, per_epoch: int = 10, pred_color=(0, 0, 255),
                 teacher_color=(0, 255, 0)):
        self._model = model
        self._dataset = dataset
        self._pred_color = pred_color
        self._teacher_color = teacher_color
        self._per_epoch = per_epoch
        self._out_dir = out_dir

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        image, teacher_bboxes = self._dataset[random_image_index]
        assert isinstance(image, torch.Tensor), "Expected image type is Tensor."
        predict_result = self._model.predict(image.view((1,) + image.shape))[0]
        image = image.detach().cpu().numpy() * 255
        image = image.transpose(1, 2, 0)

        for class_index in range(len(predict_result)):
            if predict_result[class_index] is None or len(predict_result[class_index]) == 0:
                continue
            image = draw_bounding_boxes(image, predict_result[class_index], is_bbox_norm=False, color=self._pred_color)
        image = draw_bounding_boxes(image, teacher_bboxes, is_bbox_norm=True, color=self._teacher_color)
        cv2.imwrite(f"{self._out_dir}/result_{epoch + 1}.png", image)


def draw_bounding_boxes(origin_image: np.ndarray, bounding_boxes: Iterable[Iterable[int]], is_bbox_norm=False,
                        thickness=1,
                        color=(0, 0, 255)):
    image = origin_image.copy()
    height, width = image.shape[:2]
    for bounding_box in bounding_boxes:
        if len(bounding_box) < 4:
            continue
        x_min, y_min, x_max, y_max = bounding_box[:4]
        if is_bbox_norm:
            x_min *= width
            x_max *= width
            y_min *= height
            y_max *= height
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
    return image
