from pathlib import Path
import cv2
import tqdm
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.data.transforms import AlbumentationsImageWrapperTransform
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel
from deepext.models.segmentation import CustomShelfNet
from deepext.utils import try_cuda
from deepext.data.dataset import ImageOnlyDataset

load_dotenv(".env")

weight_path = os.environ.get("SEGMENTATION_MODEL_WEIGHT_PATH")
dataset_dir = os.environ.get("SEGMENTATION_TEST_IMAGES_PATH")
out_dir = os.environ.get("OUT_DIR_PATH")
img_size = (int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT")))
n_classes = int(os.environ.get("N_CLASSES"))

if not Path(out_dir).exists():
    Path(out_dir).mkdir()

transforms = AlbumentationsImageWrapperTransform(A.Compose([
    A.Resize(img_size[0], img_size[1]),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=dataset_dir, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model: SegmentationModel = try_cuda(
    CustomShelfNet(n_classes=n_classes, out_size=img_size, backbone=BackBoneKey.RESNET_18))
model.load_weight(weight_path)
print("Model loaded")

for i, image in enumerate(tqdm.tqdm(dataset)):
    result_idx_array, result_image = model.calc_segmentation_image(image)
    result_image = cv2.resize(result_image, dataset.current_image_size())
    cv2.imwrite(f"{out_dir}/result_{i}.jpg", result_image)
