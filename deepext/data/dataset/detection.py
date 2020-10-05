from typing import List, Union, Dict, Tuple
from torch.utils.data import Dataset
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


class AdjustDetectionTensorCollator:
    """
    N * 5(x_min, y_min, x_max, y_max, class label)の1アノテーションデータをTensorに変換する.
    すべてのBounding boxを0・-1埋めで固定長にする.
    """

    def __init__(self, padding_val=-1):
        self._padding_val = padding_val

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        return [torch.stack(images, dim=0), self._resize_batch_bbox(targets)]

    def _resize_batch_bbox(self, batch_bboxes: List[List[List[Union[float, int]]]]):
        """
        :param batch_bboxes: Batch size * N * 5(x_min, y_min, x_max, y_max, class label)
        :return: Batch size * max_bbox * 5
        """
        max_bbox = max(list(map(lambda bboxes: len(bboxes), batch_bboxes)))
        new_batch_bboxes = np.ones([len(batch_bboxes), max_bbox, 5]) * self._padding_val
        for i, bboxes in enumerate(batch_bboxes):
            for j in range(len(batch_bboxes[i])):
                new_batch_bboxes[i, j, 0] = batch_bboxes[i][j][0]
                new_batch_bboxes[i, j, 1] = batch_bboxes[i][j][1]
                new_batch_bboxes[i, j, 2] = batch_bboxes[i][j][2]
                new_batch_bboxes[i, j, 3] = batch_bboxes[i][j][3]
                new_batch_bboxes[i, j, 4] = batch_bboxes[i][j][4]
        return new_batch_bboxes


class VOCAnnotationTransform:
    """
    VOC形式のXMLデータを加工するTransform
    """
    points = ["xmin", "ymin", "xmax", "ymax"]

    def __init__(self, class_index_dict: Dict, size: Tuple[int, int] = None, ignore_labels: List[str] = None):
        self._size = size
        self._class_index_dict = class_index_dict
        self._ignore_labels = ignore_labels or []

    def __call__(self, target: ET.Element or Dict):
        assert isinstance(target, ET.Element) or isinstance(target, dict)
        is_target_dict = isinstance(target, dict)
        result = []

        width = int(target["annotation"]["size"]["width"]) if is_target_dict else int(target.find("size/width").text)
        height = int(target["annotation"]["size"]["height"]) if is_target_dict else int(target.find("size/height").text)

        adjust_width_rate = self._size[1] / width if self._size is not None else 1.
        adjust_height_rate = self._size[0] / height if self._size is not None else 1.

        obj_list = target["annotation"]["object"] if is_target_dict else target.findall("object")
        obj_list = [obj_list, ] if not isinstance(obj_list, list) else obj_list

        for obj in obj_list:
            class_name = obj["name"] if is_target_dict else obj.find("name").text
            bbox_obj = obj["bndbox"] if is_target_dict else obj.find("bndbox")
            bbox = []
            for i, point in enumerate(self.points):
                coordinate = int(bbox_obj[point] if is_target_dict else bbox_obj.find(point).text) - 1
                coordinate = coordinate * adjust_width_rate if i % 2 == 0 else coordinate * adjust_height_rate
                bbox.append(coordinate)
            class_index = self._class_index_dict.get(class_name)
            if class_index is None or class_name in self._ignore_labels:  # Except not exist class.
                continue
            result.append(bbox + [class_index, ])
        return result


class VOCDataset(Dataset):
    def __init__(self, image_dir_path: str, annotation_dir_path: str, transforms, class_index_dict: Dict[str, int],
                 valid_suffixes: List[str] = None):
        """
        :param image_dir_path: Directory path of images.
        :param annotation_dir_path: Directory path of VOC format XML files.
        :param transforms: LabelAndDataTransforms or Albumentations.Compose
        :param class_index_dict: {Class name: Class label num, ...}
        :param valid_suffixes:
        """
        self.transforms = transforms
        image_dir = Path(image_dir_path)

        self._voc_transform = VOCAnnotationTransform(class_index_dict=class_index_dict, size=None)

        if valid_suffixes is None:
            valid_suffixes = ["*.png", "*.jpg", "*.jpeg", ]
        self.image_path_ls = []
        for suffix in valid_suffixes:
            self.image_path_ls += list(image_dir.glob(suffix))

        self.annotation_dir = Path(annotation_dir_path)
        self.image_path_ls.sort()

    def __call__(self, idx: int):
        image_path = self.image_path_ls[idx]
        image = Image.open(str(image_path))

        annotation_path = self.annotation_dir.joinpath(f"{image_path.stem}.xml")
        annotation_node = ET.parse(str(annotation_path)).getroot()
        annotation = self._voc_transform(annotation_node)

        return self.transforms(image, annotation)
