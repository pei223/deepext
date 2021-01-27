import numpy as np
from typing import List
from torch.utils.data import Dataset
import tqdm
from pathlib import Path

from . import ModelCallback
from ...utils import try_cuda, image_utils
from ...models.base import AttentionClassificationModel, ClassificationModel


class GenerateAttentionMapCallback(ModelCallback):
    def __init__(self, output_dir: str, per_epoch: int, dataset: Dataset, model: AttentionClassificationModel,
                 label_names: List[str], apply_all_images=False):
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
        self._apply_all_images = apply_all_images
        if not Path(self._out_dir).exists():
            Path(self._out_dir).mkdir()

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        if self._apply_all_images:
            for i, (img_tensor, label) in enumerate(self._dataset):
                pred_label, attention_image = self._model.predict_label_and_heatmap(img_tensor)
                image_utils.cv_to_pil(attention_image).save(self._image_path(epoch, pred_label, label, f"data{i}_"))
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, label = self._dataset[random_image_index]
        pred_label, attention_image = self._model.predict_label_and_heatmap(img_tensor)

        # NOTE OpenCVで保存した方が速いが、日本語対応してないからこうしてる
        image_utils.cv_to_pil(attention_image).save(self._image_path(epoch, pred_label, label))

    def _image_path(self, epoch: int, pred_label: int, label: int, prefix=""):
        return f"{self._out_dir}/{prefix}epoch{epoch + 1}_T_{self._label_names[label]}_P_{self._label_names[pred_label]}.png"


class CSVClassificationResultCallback(ModelCallback):
    def __init__(self, out_filepath: str, per_epoch: int, dataset: Dataset, model: ClassificationModel,
                 label_names: List[str]):
        self._model = model
        self._out_filepath = out_filepath
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._label_names = label_names

    def __call__(self, epoch):
        if (epoch + 1) % self._per_epoch != 0:
            return
        with open(self._out_filepath, "w") as file:
            file.write("number,predict class,teacher class,predict name,teacher name\n")
            for i, datum in enumerate(tqdm.tqdm(self._dataset)):
                image, label = datum
                pred = self._model.predict_label(image)
                file.write(f"{i},{pred},{label},{self._label_names[pred]},{self._label_names[label]}\n")
