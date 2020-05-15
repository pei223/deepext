import torch
import torch.optim as optim
import numpy as np
import copy

from ...base.base_model import BaseModel
from .m2det_lib.m2det import build_net
from .m2det_lib.utils.core import init_net, image_forward
from .m2det_lib.layers.functions import PriorBox, Detect
from .m2det_lib.layers.modules import MultiBoxLoss
from .m2det_lib.data import mk_anchors
from ...utils import try_cuda


class M2Det(BaseModel):
    def __init__(self, num_classes, input_size=512, weight_path=None, score_threshold=0.2):
        super().__init__()
        arg_init_net = True
        arg_pretrained = None
        lr = [0.004, 0.002, 0.0004, 0.00004, 0.000004]
        bkg_label = 0

        net = build_net(
            num_classes,
            'train',
            size=input_size,  # Only 320, 512, 704 and 800 are supported
            config=None)
        init_net(net, arg_init_net, arg_pretrained, weight_path)

        self._model = try_cuda(copy.deepcopy(net))
        self._optimizer = optim.SGD(
            net.parameters(),
            lr=lr[0],
            momentum=0.9,
            weight_decay=0.0005
        )
        self._criterion = MultiBoxLoss(
            num_classes,
            overlap_thresh=0.5,
            prior_for_matching=True,
            bkg_label=bkg_label,
            neg_mining=True,
            neg_pos=3,
            neg_overlap=0.5,
            encode_target=False
        )
        anchors = mk_anchors(
            input_size,
            input_size,
            [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            [8, 16, 32, 64, 128, 256],
        )
        priorbox = PriorBox(anchors)
        with torch.no_grad():
            self._priors = priorbox.forward()
            self._priors = self._priors.cuda()
        self._detector = Detect(num_classes, bkg_label, anchors)
        self._score_threshold = score_threshold

    def train_batch(self, inputs: torch.Tensor, teachers: torch.Tensor):
        self._model.train()

        images = torch.Tensor(inputs).cuda().float()
        annotations = torch.Tensor(teachers).cuda()

        out = self._model(images)
        self._optimizer.zero_grad()
        try:
            loss_l, loss_c = self._criterion(out, self._priors, annotations)
        except:
            import traceback
            traceback.print_exc()
            return -1
        loss = loss_l + loss_c
        loss.backward()
        self._optimizer.step()
        print(loss.item())
        return float(loss)

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        self._model.eval()

        scores = []
        boxes = []
        labels = []
        with torch.no_grad():
            for i in range(inputs.shape[0]):
                image = torch.Tensor(inputs[0]).cuda().float().unsqueeze(0)
                box, score = image_forward(image, self._model, True, self._priors, self._detector)
                label = -1  # label.cpu().numpy() 応急処置
                scores.append(score)
                labels.append(label)
                boxes.append(box)
        return np.array(scores), np.array(boxes), np.array(labels)

    def save_weight(self, save_path):
        torch.save(self._model.state_dict(), save_path)

    def load_weight(self, weight_path: str):
        torch.load(self._model, weight_path)

    def get_optimizer(self):
        return self._optimizer

    def get_model_config(self):
        return {}
