from pathlib import Path
import cv2
from deepext.models.base import DetectionModel
from deepext.models.object_detection import EfficientDetector
from deepext.utils import try_cuda
import tqdm

# TODO write path
weight_path = ""
dataset_dir = ""
out_dir = ""

if not Path(out_dir).exists():
    Path(out_dir).mkdir()

img_size = (512, 512)
n_classes = 20
# TODO write labels
label_names = []

# TODO Choose model, parameters.
model: DetectionModel = try_cuda(
    EfficientDetector(num_classes=n_classes, network="efficientdet-d0"))
model.load_weight(weight_path)

files = list(Path(dataset_dir).glob("*.jpg")) + list(Path(dataset_dir).glob("*.png")) + list(
    Path(dataset_dir).glob("*.jpg"))
for file_path in tqdm.tqdm(files):
    image = cv2.imread(str(file_path))
    image = cv2.resize(image, img_size)
    result_image = model.calc_detection_image(image, label_names=label_names, require_normalize=True)
    cv2.imwrite(f"{out_dir}/{file_path.name}", result_image)
