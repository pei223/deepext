from pathlib import Path
import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.layers.loss import SegmentationFocalLoss
from deepext.data.dataset import IndexImageDataset
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel
from deepext.models.segmentation import UNet, ResUNet, CustomShelfNet, ShelfNetRealtime
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, GenerateSegmentationImageCallback
from deepext.data.transforms import AlbumentationsSegmentationWrapperTransform
from deepext.metrics.segmentation import *
from deepext.utils import *

load_dotenv(".env")

# File/Directory path
train_images_dir = os.environ.get("SEGMENTATION_TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("SEGMENTATION_TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("SEGMENTATION_TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("SEGMENTATION_TEST_ANNOTATIONS_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
saved_weights_dir = os.environ.get("SAVED_WEIGHTS_DIR_PATH")
load_weight_path = os.environ.get("SEGMENTATION_MODEL_WEIGHT_PATH")
label_file_path = os.environ.get("SEGMENTATION_LABEL_FILE_PATH")
# Model params
image_size = (int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT")))
n_classes = int(os.environ.get("N_CLASSES"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

if not Path(progress_dir).exists():
    Path(progress_dir).mkdir()
if not Path(saved_weights_dir).exists():
    Path(saved_weights_dir).mkdir()

label_names = []
with open(label_file_path, "r") as file:
    for line in file:
        label_names.append(line.replace("\n", ""))

# TODO Learning detail params
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = A.Compose([
    A.HorizontalFlip(),
    A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(image_size[1] / 5), max_width=int(image_size[0] / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
])
train_transforms = AlbumentationsSegmentationWrapperTransform(train_transforms, class_num=n_classes,
                                                              ignore_indices=ignore_indices)

test_transforms = A.Compose([
    A.Resize(image_size[0], image_size[1]),
    ToTensorV2(),
])
test_transforms = AlbumentationsSegmentationWrapperTransform(test_transforms, class_num=n_classes,
                                                             ignore_indices=ignore_indices)

# dataset/dataloader
train_dataset = IndexImageDataset.create(image_dir_path=train_images_dir, index_image_dir_path=train_annotations_dir,
                                         transforms=train_transforms)
test_dataset = IndexImageDataset.create(image_dir_path=test_images_dir, index_image_dir_path=test_annotations_dir,
                                        transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# TODO Model detail params
voc_focal_loss = SegmentationFocalLoss()
model: SegmentationModel = try_cuda(CustomShelfNet(n_classes=n_classes, out_size=image_size, loss_func=voc_focal_loss))
if load_weight_path and load_weight_path != "":
    model.load_weight(load_weight_path)

# TODO Train detail params
# Metrics/Callbacks
callbacks = [ModelCheckout(per_epoch=int(epoch / 5), model=model, our_dir=saved_weights_dir),
             GenerateSegmentationImageCallback(output_dir=progress_dir, per_epoch=5, model=model,
                                               dataset=test_dataset)]
metric_ls = [SegmentationIoUByClasses(label_names), SegmentationRecallPrecision(label_names)]
metric_for_graph = SegmentationIoUByClasses(label_names, val_key=MetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=0,
                                                    metric_for_graph=metric_for_graph,
                                                    save_filepath="segmentation_learning_curve.png")

# Training.
Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(data_loader=train_dataloader,
                                                                        test_dataloader=test_dataloader,
                                                                        epochs=epoch, callbacks=callbacks,
                                                                        lr_scheduler_func=lr_scheduler,
                                                                        metric_ls=metric_ls,
                                                                        calc_metrics_per_epoch=5)
