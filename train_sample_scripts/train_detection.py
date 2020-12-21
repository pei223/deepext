from pathlib import Path
import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.data.dataset import VOCDataset
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import DetectionModel
from deepext.models.object_detection import EfficientDetector
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, VisualizeRandomObjectDetectionResult
from deepext.data.transforms import AlbumentationsDetectionWrapperTransform
from deepext.data.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator
from deepext.metrics.object_detection import *
from deepext.utils import try_cuda

load_dotenv(".env")

# File/Directory path
train_images_dir = os.environ.get("DETECTION_TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("DETECTION_TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("DETECTION_TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("DETECTION_TEST_ANNOTATIONS_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
saved_weights_dir = os.environ.get("SAVED_WEIGHTS_DIR_PATH")
load_weight_path = os.environ.get("DETECTION_MODEL_WEIGHT_PATH")
label_file_path = os.environ.get("DETECTION_LABEL_FILE_PATH")
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

label_names, class_index_dict = [], {}
i = 0
with open(label_file_path, "r") as file:
    for line in file:
        label_name = line.replace("\n", "")
        label_names.append(label_name)
        class_index_dict[label_name] = i
        i += 1

# TODO Learning detail params
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = [
    A.HorizontalFlip(),
    A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(image_size[1] / 5), max_width=int(image_size[0] / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
]
train_transforms = AlbumentationsDetectionWrapperTransform(train_transforms,
                                                           annotation_transform=VOCAnnotationTransform(
                                                               class_index_dict))

test_transforms = [
    A.Resize(image_size[0], image_size[1]),
    ToTensorV2(),
]
test_transforms = AlbumentationsDetectionWrapperTransform(test_transforms,
                                                          annotation_transform=VOCAnnotationTransform(class_index_dict))

# dataset/dataloader
train_dataset = VOCDataset(image_dir_path=train_images_dir, transforms=train_transforms,
                           class_index_dict=class_index_dict)
test_dataset = VOCDataset(image_dir_path=test_images_dir, transforms=test_transforms, class_index_dict=class_index_dict)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=AdjustDetectionTensorCollator())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=AdjustDetectionTensorCollator())

# TODO Model detail params
model: DetectionModel = try_cuda(EfficientDetector(num_classes=n_classes, lr=lr,
                                                   network=f"efficientdet-d0", score_threshold=0.5))
if load_weight_path and load_weight_path != "":
    model.load_weight(load_weight_path)

# TODO Train detail params
# Metrics/Callbacks
callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir=saved_weights_dir),
             VisualizeRandomObjectDetectionResult(model, image_size, test_dataset, per_epoch=1,
                                                  out_dir=progress_dir, label_names=label_names)]
metric_ls = [DetectionIoUByClasses(label_names), RecallAndPrecision(label_names)]
metric_for_graph = DetectionIoUByClasses(label_names, val_key=MetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=10,
                                                    metric_for_graph=metric_for_graph,
                                                    save_filepath="detection_learning_curve.png")

# Training.
Trainer(model).fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                   epochs=epoch, callbacks=callbacks, lr_scheduler_func=lr_scheduler, metric_ls=metric_ls,
                   calc_metrics_per_epoch=5, learning_curve_visualizer=learning_curve_visualizer)
