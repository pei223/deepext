from typing import List, Union, Dict, Tuple

import numpy as np
import torch


class ObjectDetectionCollator:
    """
    N * 5(x_min, y_min, x_max, y_max, class label)の1アノテーションデータをTensorに変換する.
    すべてのBounding boxを0・-1埋めで固定長にする.
    """

    def __init__(self, padding_val=-1):
        self._padding_val = padding_val

    def __call__(self, batch):
        images = []
        targets = []
        for sample in batch:
            image, target = sample
            images.append(image)
            targets.append(target)
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

    def __init__(self, class_index_dict: Dict, size: Tuple[int, int]):
        self._height, self._width = size
        self._class_index_dict = class_index_dict

    def __call__(self, target):
        result = []
        width, height = int(target["annotation"]["size"]["width"]), int(target["annotation"]["size"]["height"])
        points = ["xmin", "ymin", "xmax", "ymax"]
        obj_list = target["annotation"]["object"]
        adjust_width_rate = self._width / width
        adjust_height_rate = self._height / height

        if not isinstance(obj_list, list):
            obj_list = [obj_list, ]
        for obj in obj_list:
            class_name = obj["name"]
            bbox_obj = obj["bndbox"]
            bbox = []
            for i, point in enumerate(points):
                coordinate = int(bbox_obj[point]) - 1
                coordinate = coordinate * adjust_width_rate if i % 2 == 0 else coordinate * adjust_height_rate
                bbox.append(coordinate)
            class_index = self._class_index_dict[class_name]
            bbox.append(class_index)
            result.append(bbox)
        return result
