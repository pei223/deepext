import torch
import torch.optim as optim
import numpy as np

from ....models.base.base_model import BaseModel
from .ssd_lib.ssd import build_ssd
from .ssd_lib.layers.modules.multibox_loss import MultiBoxLoss
from ....utils.tensor_util import try_cuda


class SSD(BaseModel):
    def __init__(self, num_classes, input_size=512, lr=1e-4):
        super().__init__()
        self._model = build_ssd(size=input_size, num_classes=num_classes, phase="train")

        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4)
        self._criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5,
                                       False, torch.cuda.is_available())

    def train_batch(self, inputs: torch.Tensor, teachers: torch.Tensor):
        self._model.train()
        self._model.phase = "train"

        images = torch.Tensor(inputs).cuda().float()
        annotations = torch.Tensor(teachers).cuda()

        out = self._model(images)
        print(out[0].shape, out[1].shape, annotations.shape)
        self._optimizer.zero_grad()
        loss_l, loss_c = self._criterion(out, annotations)
        loss = loss_l + loss_c
        loss.backward()
        self._optimizer.step()
        print(loss.item())
        return float(loss)

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        self._model.eval()
        self._model.phase = "test"

        scores = []
        boxes = []
        labels = []
        with torch.no_grad():
            for i in range(inputs.shape[0]):
                image = try_cuda(inputs[0]).float().unsqueeze(0)
                result = self._model(image)
                print(result[0].shape, result[1].shape, result[2].shape)
                detections = result.data
                scale = torch.Tensor([image.shape[1], image.shape[0],
                                      image.shape[1], image.shape[0]])
                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] >= 0.6:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        coords = (pt[0], pt[1], pt[2], pt[3])

                        j += 1

    def save_weight(self, save_path):
        torch.save(self._model.state_dict(), save_path)

    def load_weight(self, weight_path: str):
        torch.load(self._model, weight_path)

    def get_optimizer(self):
        return self._optimizer

    def get_model_config(self):
        return {}
