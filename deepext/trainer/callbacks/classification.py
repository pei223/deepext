import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from ...models.classification import AttentionBranchNetwork


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
