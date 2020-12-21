from pathlib import Path
import cv2
import tqdm
from dotenv import load_dotenv
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import ClassificationModel
from deepext.models.classification import AttentionBranchNetwork, MobileNetV3
from deepext.utils import try_cuda
from deepext.data.dataset import ImageOnlyDataset
from deepext.data.transforms import AlbumentationsImageWrapperTransform

load_dotenv(".env")

weight_path = os.environ.get("CLASSIFICATION_MODEL_WEIGHT_PATH")
dataset_dir = os.environ.get("CLASSIFICATION_TEST_IMAGES_PATH")
out_file_path = os.environ.get("CLASSIFICATION_OUT_FILE_PATH")
label_file_path = os.environ.get("CLASSIFICATION_LABEL_FILE_PATH")
img_size = (int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT")))
n_classes = int(os.environ.get("N_CLASSES"))

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
model: ClassificationModel = try_cuda(MobileNetV3(num_classes=n_classes))
model.load_weight(weight_path)
print("Model loaded")

with open(out_file_path, "w") as file:
    file.write(f"filepath,result label\n")
    for i, image in enumerate(tqdm.tqdm(dataset)):
        label = model.predict_label(image)
        file.write(f"{dataset.current_file_path()},{label_names[label]}\n")
