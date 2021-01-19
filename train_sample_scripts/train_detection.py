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
from deepext.data.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator, MultiVOCDatasetFactory
from deepext.metrics.object_detection import *
from deepext.utils import try_cuda
from deepext.utils.dataset_util import create_label_list_and_dict, create_train_test_indices

load_dotenv("envs/detection.env")

# File/Directory path
train_images_dir = os.environ.get("TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("TEST_ANNOTATIONS_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")

load_weight_path = os.environ.get("MODEL_WEIGHT_PATH")
saved_weights_dir = os.environ.get("SAVED_WEIGHTS_DIR_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
# Model params
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

if not Path(progress_dir).exists():
    Path(progress_dir).mkdir()
if not Path(saved_weights_dir).exists():
    Path(saved_weights_dir).mkdir()

label_names, class_index_dict = create_label_list_and_dict(label_file_path)

# TODO Learning detail params
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = AlbumentationsDetectionWrapperTransform([
    A.HorizontalFlip(),
    A.RandomResizedCrop(width=width, height=height, scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(height / 5), max_width=int(width / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
], annotation_transform=VOCAnnotationTransform(class_index_dict))

test_transforms = AlbumentationsDetectionWrapperTransform([
    A.Resize(width=width, height=height),
    ToTensorV2(),
], annotation_transform=VOCAnnotationTransform(class_index_dict))

# dataset/dataloader
if test_images_dir == "":
    data_len = int(os.environ.get("DATA_LEN"))
    test_ratio = float(os.environ.get("TEST_RATIO"))
    train_indices, test_indices = create_train_test_indices(data_len, test_ratio)
    train_dataset, test_dataset = MultiVOCDatasetFactory(image_dir_path=train_images_dir,
                                                         annotation_dir_path=train_annotations_dir,
                                                         train_transforms=train_transforms,
                                                         test_transforms=test_transforms,
                                                         class_index_dict=class_index_dict) \
        .create_train_test(train_indices, test_indices)
else:
    train_dataset = VOCDataset.create(image_dir_path=train_images_dir, annotation_dir_path=train_annotations_dir,
                                      transforms=train_transforms, class_index_dict=class_index_dict)
    test_dataset = VOCDataset.create(image_dir_path=test_images_dir, transforms=test_transforms,
                                     annotation_dir_path=test_annotations_dir, class_index_dict=class_index_dict)

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
callbacks = [ModelCheckout(per_epoch=int(epoch / 5), model=model, our_dir=saved_weights_dir),
             VisualizeRandomObjectDetectionResult(model, (height, width), test_dataset, per_epoch=5,
                                                  out_dir=progress_dir, label_names=label_names)]
metric_ls = [DetectionIoUByClasses(label_names), RecallAndPrecision(label_names)]
metric_for_graph = DetectionIoUByClasses(label_names, val_key=DetailMetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=10,
                                                    save_filepath="detection_learning_curve.png")

# Training.
Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(train_data_loader=train_dataloader,
                                                                        test_data_loader=test_dataloader,
                                                                        epochs=epoch, callbacks=callbacks,
                                                                        lr_scheduler_func=lr_scheduler,
                                                                        metric_for_graph=metric_for_graph,
                                                                        metric_ls=metric_ls,
                                                                        calc_metrics_per_epoch=5)
