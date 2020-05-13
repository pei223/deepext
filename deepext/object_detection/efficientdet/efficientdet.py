import torch.optim as optim
import torch
import numpy as np

from deepext.base.base_model import BaseModel
from .efficientdet_lib.models.efficientdet import EfficientDet
from .efficientdet_lib.utils import EFFICIENTDET, get_state_dict
from deepext.utils.tensor_util import try_cuda


class EfficientDetector(BaseModel):

    def __init__(self, num_classes, network='efficientdet-d0', lr=1e-4, score_threshold=0.2, max_detections=100):
        super().__init__()
        self._model = EfficientDet(num_classes=num_classes,
                                   network=network,
                                   W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                                   D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                                   D_class=EFFICIENTDET[network]['D_class']).cuda()
        self._num_classes = num_classes
        self._network = network
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, patience=3, verbose=True)
        self._score_threshold = score_threshold
        self._max_detections = max_detections

    def train_batch(self, inputs, teachers):
        self._model.train()
        self._model.is_training = True
        self._model.freeze_bn()
        self._optimizer.zero_grad()

        images = try_cuda(torch.Tensor(inputs)).float()
        annotations = try_cuda(torch.Tensor(teachers))
        classification_loss, regression_loss = self._model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        print(classification_loss.item(), regression_loss.item())
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            return 0.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
        self._optimizer.step()
        self._optimizer.zero_grad()
        # self._scheduler.step(loss.item())
        return float(loss)

    def predict(self, inputs):
        self._model.eval()
        self._model.is_training = False
        batch_size = inputs.shape[0]
        all_detections = [[None for i in range(self._num_classes)] for j in range(batch_size)]

        with torch.no_grad():
            for i in range(batch_size):
                image = inputs[i].float().unsqueeze(0)
                scores, labels, boxes = self._model(try_cuda(image))
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()
                indices = np.where(scores > self._score_threshold)[0]
                if indices.shape[0] > 0:
                    scores = scores[indices]
                    scores_sort = np.argsort(-scores)[:self._max_detections]
                    # select detections
                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                    image_detections = np.concatenate([
                        image_boxes,
                        np.expand_dims(image_scores, axis=1),
                        np.expand_dims(image_labels, axis=1)
                    ], axis=1)

                    # copy detections to all_detections
                    for label in range(self._num_classes):
                        all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
        return np.asarray(all_detections)

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
            'network': self._network,
            'state_dict': self._model.state_dict()
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        checkpoint = torch.load(weight_path)
        params = checkpoint['parser']
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params.num_class))
        print('Network: {}'.format(params.network))
        self._model.load_state_dict(checkpoint['state_dict'])

    def get_model_config(self):
        return {}

    def get_optimizer(self):
        return self._optimizer
