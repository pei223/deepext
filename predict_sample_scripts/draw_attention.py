from pathlib import Path
import cv2
from dotenv import load_dotenv
import os, tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import AttentionClassificationModel
from deepext.models.classification import AttentionBranchNetwork
from deepext.data.dataset import ImageOnlyDataset
from deepext.data.transforms import AlbumentationsImageWrapperTransform
from deepext.utils import try_cuda

load_dotenv(".env")

weight_path = os.environ.get("ATTENTION_CLASSIFICATION_MODEL_WEIGHT_PATH")
dataset_dir = os.environ.get("ATTENTION_CLASSIFICATION_TEST_IMAGES_PATH")
label_file_path = os.environ.get("CLASSIFICATION_LABEL_FILE_PATH")
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
model: AttentionClassificationModel = try_cuda(
    AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18))
model.load_weight(weight_path)
print("Model loaded")

for i, image in enumerate(tqdm.tqdm(dataset)):
    label, result_image = model.predict_label_and_heatmap(image)
    result_image = cv2.resize(result_image, dataset.current_image_size())
    cv2.imwrite(f"{out_dir}/{label_names[label]}_{i}.jpg", result_image)
