import numpy as np
from typing import List
from torch.utils.data import Dataset

from ...utils import try_cuda, image_utils
from ...models.classification import AttentionBranchNetwork


class GenerateAttentionMapCallback:
    def __init__(self, output_dir: str, per_epoch: int, dataset: Dataset, model: AttentionBranchNetwork,
                 label_names: List[str]):
        """
        :param output_dir: 経過出力先ディレクトリ
        :param per_epoch: 何エポックごとにコールバックを実行するか
        :param dataset: 推論経過に使用するDataset
        :param model: AttentionBranchNetwork系モデル
        :param label_names: ラベル名がクラス順に並んでいるリスト
        """
        self._out_dir, self._per_epoch, self._dataset = output_dir, per_epoch, dataset
        self._model = model
        self._label_names = label_names

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        self._model.eval()
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, label = self._dataset[random_image_index]
        img_tensor = try_cuda(img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]))
        perception_pred, attention_pred, attention_map = self._model(img_tensor)
        pred_label = perception_pred.argmax(-1).item()

        # 0~1のTensorを0~255に変換
        origin_img = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        heatmap_img = attention_map[0][0].cpu().detach().numpy()
        origin_img, heatmap_img = (origin_img * 255).astype("uint8"), (heatmap_img * 255).astype("uint8")

        blended_img = image_utils.combine_heatmap(origin_img, heatmap_img, origin_alpha=0.5)

        # NOTE OpenCVで保存した方が速いが、日本語対応してないからこうしてる
        image_utils.cv_to_pil(blended_img).save(self._image_path(epoch, pred_label, label))

    def _image_path(self, epoch: int, pred_label: int, label: int):
        return f"{self._out_dir}/epoch{epoch + 1}_T_{self._label_names[label]}_P_{self._label_names[pred_label]}.png"
