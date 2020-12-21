from pathlib import Path
import cv2
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import ClassificationModel
from deepext.models.classification import MobileNetV3
from deepext.utils import try_cuda
import tqdm

# TODO write path
weight_path = ""
dataset_dir = ""
out_dir = ""

if not Path(out_dir).exists():
    Path(out_dir).mkdir()

img_size = (512, 512)
n_classes = 21
# TODO
label_names = []

# TODO Choose model, parameters.
model: ClassificationModel = try_cuda(
    MobileNetV3(num_classes=n_classes))
model.load_weight(weight_path)

files = list(Path(dataset_dir).glob("*.jpg")) + list(Path(dataset_dir).glob("*.png")) + list(
    Path(dataset_dir).glob("*.jpg"))
for file_path in tqdm.tqdm(files):
    image = cv2.imread(str(file_path))
    image = cv2.resize(image, img_size)
    result_image, label = model.predict_label(image, require_normalize=True)
    cv2.imwrite(f"{out_dir}/{label_names[label]}_{file_path.name}", result_image)
