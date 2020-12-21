from pathlib import Path
import cv2
import os, tqdm
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.models.base import DetectionModel
from deepext.models.object_detection import EfficientDetector
from deepext.data.dataset import ImageOnlyDataset
from deepext.data.transforms import AlbumentationsImageWrapperTransform
from deepext.utils import try_cuda

load_dotenv(".env")

weight_path = os.environ.get("DETECTION_MODEL_WEIGHT_PATH")
dataset_dir = os.environ.get("DETECTION_TEST_IMAGES_PATH")
label_file_path = os.environ.get("DETECTION_LABEL_FILE_PATH")
out_dir = os.environ.get("OUT_DIR_PATH")
img_size = (int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT")))
n_classes = int(os.environ.get("N_CLASSES"))

if not Path(out_dir).exists():
    Path(out_dir).mkdir()

label_names = []
with open(label_file_path, "r") as file:
    for line in file:
        label_names.append(line.replace("\n", ""))

transforms = AlbumentationsImageWrapperTransform(A.Compose([
    A.Resize(img_size[0], img_size[1]),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=dataset_dir, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model: DetectionModel = try_cuda(
    EfficientDetector(num_classes=n_classes, network="efficientdet-d0"))
model.load_weight(weight_path)
print("Model loaded")

for i, image in enumerate(tqdm.tqdm(dataset)):
    result_bboxes, result_image = model.calc_detection_image(image, label_names=label_names)
    result_image = cv2.resize(result_image, dataset.current_image_size())
    cv2.imwrite(f"{out_dir}/result_{i}.jpg", result_image)
