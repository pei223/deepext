import argparse
from torchvision.transforms import Resize, Compose
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext.models.base import BaseModel
from deepext.models.object_detection import EfficientDetector, M2Det
from deepext.trainer import Trainer, LearningCurveVisualizer
from deepext.trainer.callbacks import LearningRateScheduler, ModelCheckout, VisualizeRandomObjectDetectionResult
from deepext.metrics.object_detection import *
from deepext.metrics import MetricKey
from deepext.data.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator
from deepext.utils import *

from util import DataSetSetting

# NOTE モデル・データセットはここを追加
MODEL_EFFICIENT_DET = "efficientdet"
MODEL_M2DET = "m2det"
MODEL_TYPES = [MODEL_EFFICIENT_DET, MODEL_M2DET]
DATASET_VOC2012 = "voc2012"
DATASET_VOC2007 = "voc2007"
DATASET_TYPES = [DATASET_VOC2007, DATASET_VOC2012]
settings = [DataSetSetting(dataset_type=DATASET_VOC2012, size=(512, 512), n_classes=20,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"]),
            DataSetSetting(dataset_type=DATASET_VOC2007, size=(512, 512), n_classes=20,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"])]


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = Compose([Resize(setting.size), ToTensor()])
    test_transforms = Compose([Resize(setting.size), ToTensor()])
    train_dataset, test_dataset = None, None
    class_index_dict = {}
    for i, label_name in enumerate(setting.label_names):
        class_index_dict[label_name] = i
    # NOTE データセットはここを追加
    if DATASET_VOC2012 == setting.dataset_type:
        train_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year='2012',
                                                          transform=train_transforms, image_set='train',
                                                          target_transform=Compose(
                                                              [VOCAnnotationTransform(class_index_dict, setting.size)]))
        test_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year='2012',
                                                         transform=test_transforms, image_set='trainval',
                                                         target_transform=Compose(
                                                             [VOCAnnotationTransform(class_index_dict, setting.size)]))
    elif DATASET_VOC2007 == setting.dataset_type:
        train_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year='2007',
                                                          transform=train_transforms, image_set='train',
                                                          target_transform=Compose(
                                                              [VOCAnnotationTransform(class_index_dict, setting.size)]))
        test_dataset = torchvision.datasets.VOCDetection(root=root_dir, download=True, year='2007',
                                                         transform=test_transforms, image_set='trainval',
                                                         target_transform=Compose(
                                                             [VOCAnnotationTransform(class_index_dict, setting.size)]))
    assert train_dataset is not None and test_dataset is not None, f"Not supported setting: {setting.dataset_type}"
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator()), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator()), train_dataset, test_dataset


def get_model(dataset_setting: DataSetSetting, model_type: str, lr: float, efficientdet_scale: int = 0):
    # NOTE モデルはここを追加
    if MODEL_EFFICIENT_DET == model_type:
        return EfficientDetector(num_classes=dataset_setting.n_classes, lr=lr,
                                 network=f"efficientdet-d{efficientdet_scale}")
    elif MODEL_M2DET == model_type:
        return M2Det(num_classes=dataset_setting.n_classes, input_size=dataset_setting.size)
    assert f"Invalid model type. Valid models is {MODEL_TYPES}"


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default=DATASET_VOC2012, help=f'Dataset type in {DATASET_TYPES}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default=MODEL_EFFICIENT_DET, help=f"Model type in {MODEL_TYPES}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientdet_scale', type=int, default=0, help="Number of scale of EfficientDet.")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DataSetSetting.from_dataset_type(settings, args.dataset)
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)
    img_size = (args.image_size, args.image_size)
    dataset_setting.set_size(img_size)

    # Fetch model and load weight.
    model: BaseModel = try_cuda(
        get_model(dataset_setting, model_type=args.model, lr=args.lr, efficientdet_scale=args.efficientdet_scale))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)
    save_weight_path = args.save_weight_path or f"./{args.model}.pth"

    # Training.
    callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir="./saved_weights")]
    if args.progress_dir:
        callbacks.append(VisualizeRandomObjectDetectionResult(model, dataset_setting.size, test_dataset, per_epoch=1,
                                                              out_dir=args.progress_dir,
                                                              label_names=dataset_setting.label_names))
    trainer = Trainer(model)
    trainer.fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                epochs=args.epoch, callbacks=callbacks,
                lr_scheduler_func=LearningRateScheduler(args.epoch),
                calc_metrics_per_epoch=5,
                learning_curve_visualizer=LearningCurveVisualizer(metric_name="mIoU",
                                                                  ignore_epoch=10,
                                                                  metric_for_graph=DetectionIoUByClasses(
                                                                      dataset_setting.label_names,
                                                                      val_key=MetricKey.KEY_AVERAGE,
                                                                  ), save_filepath="learning_curve.png"),
                metric_ls=[DetectionIoUByClasses(dataset_setting.label_names),
                           RecallAndPrecision(dataset_setting.label_names)])
    # Save weight.
    model.save_weight(save_weight_path)
