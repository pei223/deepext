import torch
import torch.optim as optim
import numpy as np

from ....models.base.detection_model import DetectionModel
from .ssd_lib.ssd import build_ssd
from .ssd_lib.layers.modules.multibox_loss import MultiBoxLoss
from ....utils.tensor_util import try_cuda


class SSD(DetectionModel):
    def __init__(self, num_classes, input_size=300, lr=1e-4, threshold=0.5, max_detection=100):
        super().__init__()
        self._model: torch.nn.Module = build_ssd(size=input_size, num_classes=num_classes, phase="train")

        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4)
        self._criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5,
                                       False, torch.cuda.is_available())
        self._model = try_cuda(self._model)
        self._input_size = input_size
        self._score_threshold = threshold
        self._max_detection = max_detection

    def train_batch(self, inputs: torch.Tensor, teachers: torch.Tensor):
        self._model.train()
        self._model.phase = "train"
        self._optimizer.zero_grad()

        images = torch.Tensor(inputs).cuda().float()
        annotations = torch.Tensor(teachers).cuda()

        out = self._model(images)
        loss_l, loss_c = self._criterion(out, annotations)
        loss = loss_l + loss_c
        loss.backward()
        self._optimizer.step()
        return float(loss)

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        self._model.eval()
        self._model.phase = "test"

        with torch.no_grad():
            inputs = try_cuda(inputs.float())
            y = self._model(inputs)
            locs, scores, _ = y

            result = []
            for i in range(inputs.shape[0]):
                loc, score = locs[i], scores[i]
                score = score.cpu().numpy()
                loc = (loc * self._input_size).cpu().numpy()
                valid_indices = np.where(score > self._score_threshold)
                valid_indices = valid_indices[0]
                if valid_indices.shape[0] <= 0:
                    result.append([])
                    continue
                score, loc = score[valid_indices], loc[valid_indices]
                sorted_scores_idx = np.argsort(-score.max(axis=-1), axis=-1)[:self._max_detection]

                score = np.argmax(score[sorted_scores_idx], axis=-1)
                loc = loc[sorted_scores_idx]
                result.append(np.concatenate([loc, score.reshape([score.shape[0], 1])], axis=-1))
        result = np.array(result)
        return result

    def save_weight(self, save_path):
        torch.save(self._model.state_dict(), save_path)

    def load_weight(self, weight_path: str):
        torch.load(self._model, weight_path)

    def get_optimizer(self):
        return self._optimizer

    def get_model_config(self):
        return {}
