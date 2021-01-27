from torch import optim, nn as nn
import torch
import numpy as np

from ....models.base.detection_model import DetectionModel
from .efficientdet_lib.models.efficientdet import EfficientDet
from .efficientdet_lib.utils import EFFICIENTDET
from ....utils.tensor_util import try_cuda

__all__ = ['EfficientDetector']


class EfficientDetector(DetectionModel):
    def __init__(self, num_classes, network='efficientdet-d0', lr=1e-4, score_threshold=0.5, max_detections=100,
                 backbone_path: str = None, backbone_pretrained=True, grad_accumulation_steps=1):
        super().__init__()
        self._model = try_cuda(EfficientDet(num_classes=num_classes,
                                            network=network,
                                            W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                                            D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                                            D_class=EFFICIENTDET[network]['D_class'], backbone_path=backbone_path,
                                            backbone_pretrained=backbone_pretrained,
                                            threshold=score_threshold))
        self._num_classes = num_classes
        self._network = network
        self._optimizer = optim.AdamW(self._model.parameters(), lr=lr)
        self._max_detections = max_detections
        self._grad_accumulation_steps = grad_accumulation_steps
        self._batch_count = 0

    def train_batch(self, inputs, teachers) -> float:
        """
        :param inputs: (batch size, channels, height, width)
        :param teachers: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        """
        self._model.train()
        self._model.is_training = True
        self._model.freeze_bn()

        images = try_cuda(inputs)
        annotations = torch.tensor(teachers) if not isinstance(teachers, torch.Tensor) else teachers
        annotations = try_cuda(annotations.float())
        classification_loss, regression_loss = self._model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        self._batch_count += 1
        if self._batch_count < self._grad_accumulation_steps:
            return float(loss)
        if bool(loss == 0):
            return 0.0
        loss.backward()
        self._batch_count = 0
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
        self._optimizer.step()
        self._optimizer.zero_grad()
        return float(loss)

    def predict(self, inputs) -> np.ndarray:
        """
        :param inputs: (batch size, channels, height, width)
        :return: (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        """
        self._model.eval()
        self._model.is_training = False
        batch_size = inputs.shape[0]

        result = []
        with torch.no_grad():
            for i in range(batch_size):
                image = inputs[i].float().unsqueeze(0)
                scores, labels, boxes = self._model(try_cuda(image))
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()
                scores_sort = np.argsort(-scores)[:self._max_detections]
                # select detections
                image_boxes = boxes[scores_sort, :]
                image_scores = scores[scores_sort]
                image_labels = labels[scores_sort]
                image_detections = np.concatenate([
                    image_boxes,
                    np.expand_dims(image_labels, axis=1),
                    np.expand_dims(image_scores, axis=1),
                ], axis=1)

                result.append(image_detections[:, :-1])
        return np.asarray(result)

    def _arrange_result(self, bboxes_by_classes):
        """
        クラスごとのbboxをまとめる
        :param bboxes_by_classes:
        :return: bboxリスト(x_min, y_min, x_max, y_max, label)
        """
        result_bboxes = []
        for label, bboxes in enumerate(bboxes_by_classes):
            if bboxes is None:
                continue
            for bbox in bboxes:
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                result_bboxes.append(bbox + [label, ])
        return result_bboxes

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
            'network': self._network,
            'state_dict': self._model.state_dict()
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        checkpoint = torch.load(weight_path)
        # params = checkpoint['parser']
        # print('The pretrained weight is loaded')
        # print('Num classes: {}'.format(params.num_class))
        # print('Network: {}'.format(params.network))
        self._model.load_state_dict(checkpoint['state_dict'])

    def get_model_config(self):
        return {
            "model": "EfficientDet",
            "scale_type": self._network,
            "classes": self._num_classes
        }

    def get_optimizer(self):
        return self._optimizer

    def get_model(self) -> nn.Module:
        return self._model
