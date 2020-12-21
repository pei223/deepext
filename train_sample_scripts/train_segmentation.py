import argparse
import torchvision
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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

voc_focal_loss = SegmentationFocalLoss()

# TODO Folder/File path
train_images_dir = ""
train_labels_dir = ""
test_images_dir = ""
test_labels_dir = ""
progress_dir = ""
load_weight_path = None

# TODO Model parameters
label_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
image_size = (512, 512)
n_classes = len(label_names)

# TODO Learning parameters
batch_size = 8
lr = 1e-4
epoch = 300
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# Data augmentation
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
train_dataset = IndexImageDataset(image_dir_path=train_images_dir, index_image_dir_path=train_labels_dir,
                                  transforms=train_transforms)
test_dataset = IndexImageDataset(image_dir_path=test_images_dir, index_image_dir_path=test_labels_dir,
                                 transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Metrics/Callbacks
model: SegmentationModel = try_cuda(CustomShelfNet(n_classes=n_classes, out_size=image_size))
if load_weight_path:
    model.load_weight(load_weight_path)

callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir="saved_weights"),
             GenerateSegmentationImageCallback(output_dir=progress_dir, per_epoch=1, model=model,
                                               dataset=test_dataset)]
metric_ls = [SegmentationIoUByClasses(label_names), SegmentationRecallPrecision(label_names)]
metric_for_graph = SegmentationIoUByClasses(label_names, val_key=MetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=0,
                                                    metric_for_graph=metric_for_graph,
                                                    save_filepath="segmentation_learning_curve.png")

# Training.
Trainer(model).fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                   epochs=epoch, callbacks=callbacks, lr_scheduler_func=lr_scheduler, metric_ls=metric_ls,
                   calc_metrics_per_epoch=5, learning_curve_visualizer=learning_curve_visualizer)
