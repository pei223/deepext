import argparse
import torchvision
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.models.base import DetectionModel
from deepext.data.transforms import AlbumentationsDetectionWrapperTransform
from deepext.models.object_detection import EfficientDetector, SSD
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, VisualizeRandomObjectDetectionResult
from deepext.metrics.object_detection import *
from deepext.metrics import MetricKey
from deepext.data.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator
from deepext.utils import *

from util import DataSetSetting


# NOTE モデル・データセットはここを追加
def build_efficientdet(dataset_setting, args):
    return EfficientDetector(num_classes=dataset_setting.n_classes, lr=args.lr,
                             network=f"efficientdet-d{args.efficientdet_scale}", score_threshold=0.5)


#
# def build_m2det(dataset_setting, args):
#     return M2Det(num_classes=dataset_setting.n_classes, input_size=dataset_setting.size)


def build_ssd(dataset_setting, args):
    return SSD(num_classes=dataset_setting.n_classes, input_size=args.image_size, lr=args.lr)


def build_voc_dataset(year: str, root_dir: str, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year=year,
                                                      transforms=train_transforms, image_set='train')
    test_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year=year,
                                                     transforms=test_transforms, image_set='trainval')
    return train_dataset, test_dataset


voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
DATASET_DICT = {
    "voc2012": DataSetSetting(dataset_type="voc2012", size=(512, 512), n_classes=20, label_names=voc_classes,
                              dataset_build_func=lambda root_dir, train_transforms, test_transforms, class_index_dict:
                              build_voc_dataset("2012", root_dir, train_transforms, test_transforms)),
    "voc2007": DataSetSetting(dataset_type="voc2007", size=(512, 512), n_classes=20, label_names=voc_classes,
                              dataset_build_func=lambda root_dir, train_transforms, test_transforms, class_index_dict:
                              build_voc_dataset("2007", root_dir, train_transforms, test_transforms)),
}
MODEL_DICT = {
    "efficientdet": build_efficientdet,
    # "m2det": build_m2det,
    "ssd": build_ssd,
}


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    class_index_dict = {}
    for i, label_name in enumerate(setting.label_names):
        class_index_dict[label_name] = i

    train_transforms = AlbumentationsDetectionWrapperTransform([
        A.HorizontalFlip(),
        A.RandomResizedCrop(setting.size[0], setting.size[1], scale=(0.5, 2.0)),
        A.CoarseDropout(max_height=int(setting.size[1] / 5), max_width=int(setting.size[0] / 5)),
        A.RandomBrightnessContrast(),
        ToTensorV2(),
    ], annotation_transform=VOCAnnotationTransform(class_index_dict))
    test_transforms = AlbumentationsDetectionWrapperTransform([
        A.Resize(setting.size[0], setting.size[1]),
        ToTensorV2(),
    ], annotation_transform=VOCAnnotationTransform(class_index_dict))

    train_dataset, test_dataset = setting.dataset_build_func(root_dir, train_transforms, test_transforms,
                                                             class_index_dict)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator()), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator()), train_dataset, test_dataset


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="voc2012", help=f'Dataset type in {list(DATASET_DICT.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="efficientdet", help=f"Model type in {list(MODEL_DICT.keys())}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientdet_scale', type=int, default=0, help="Number of scale of EfficientDet.")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DATASET_DICT.get(args.dataset)
    assert dataset_setting is not None, f"Invalid dataset type.  Valid dataset is {list(DATASET_DICT.keys())}"
    img_size = (args.image_size, args.image_size)
    dataset_setting.set_size(img_size)

    # Fetch model and load weight.
    build_model_func = MODEL_DICT.get(args.model)
    assert build_model_func is not None, f"Invalid model type. Valid models is {list(MODEL_DICT.keys())}"
    model: DetectionModel = try_cuda(build_model_func(dataset_setting, args))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)

    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)

    # Training setting.
    lr_scheduler = CosineDecayScheduler(max_lr=args.lr, max_epochs=args.epoch, warmup_epochs=0)
    callbacks = [ModelCheckout(per_epoch=int(args.epoch / 5), model=model, our_dir="saved_weights")]
    if args.progress_dir:
        callbacks.append(VisualizeRandomObjectDetectionResult(model, dataset_setting.size, test_dataset, per_epoch=5,
                                                              out_dir=args.progress_dir,
                                                              label_names=dataset_setting.label_names))
    metric_ls = [DetectionIoUByClasses(dataset_setting.label_names), RecallAndPrecision(dataset_setting.label_names)]
    metric_for_graph = DetectionIoUByClasses(dataset_setting.label_names, val_key=MetricKey.KEY_AVERAGE)
    learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=10,
                                                        metric_for_graph=metric_for_graph,
                                                        save_filepath="detection_learning_curve.png")
    # Training.
    Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(data_loader=train_dataloader,
                                                                            test_dataloader=test_dataloader,
                                                                            epochs=args.epoch,
                                                                            callbacks=callbacks, metric_ls=metric_ls,
                                                                            lr_scheduler_func=lr_scheduler,
                                                                            calc_metrics_per_epoch=10)
