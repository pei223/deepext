import argparse
import torchvision
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.layers.loss import SegmentationFocalLoss
from deepext.layers.subnetwork import BackBoneKey
from deepext.models.base import SegmentationModel
from deepext.models.segmentation import PSPNet, UNet, ResUNet, ResPSPNet, CustomShelfNet, ShelfNetRealtime, \
    ShallowShelfNet
from deepext.trainer import Trainer, LearningCurveVisualizer, LearningRateScheduler
from deepext.trainer.callbacks import ModelCheckout, GenerateSegmentationImageCallback
from deepext.data.transforms import AlbumentationsSegmentationWrapperTransform
from deepext.metrics.segmentation import *
from deepext.utils import *

from util import DataSetSetting

voc_focal_loss = SegmentationFocalLoss()


def build_pspnet(dataset_setting, args):
    if args.submodel is None:
        return PSPNet(n_classes=dataset_setting.n_classes, img_size=dataset_setting.size, lr=args.lr)
    if args.submodel == "resnet":
        return ResPSPNet(n_classes=dataset_setting.n_classes, img_size=dataset_setting.size, lr=args.lr)
    assert f"Invalid sub model type: {args.submodel}.  {args.model} Model require resnet or none."


def build_unet(dataset_setting, args):
    loss_func = voc_focal_loss if args.dataset in ["voc2007", "voc2012"] else None
    if args.submodel is None:
        return UNet(n_input_channels=3, n_output_channels=dataset_setting.n_classes, lr=args.lr, loss_func=loss_func)
    if args.submodel == "resnet":
        return ResUNet(n_input_channels=3, n_output_channels=dataset_setting.n_classes, lr=args.lr, loss_func=loss_func)
    assert f"Invalid sub model type: {args.submodel}.  {args.model} Model require resnet or none."


def build_custom_shelfnet(dataset_setting, args):
    loss_func = voc_focal_loss if args.dataset in ["voc2007", "voc2012"] else None
    return CustomShelfNet(n_classes=dataset_setting.n_classes, lr=args.lr, out_size=dataset_setting.size,
                          loss_func=loss_func, backbone=BackBoneKey.from_val(args.submodel))


def build_shallow_shelfnet(dataset_setting, args):
    return ShallowShelfNet(n_classes=dataset_setting.n_classes, lr=args.lr, out_size=dataset_setting.size,
                           loss_type="ce", backbone=args.submodel)


def build_shelfnet_realtime(dataset_setting, args):
    return ShelfNetRealtime(size=dataset_setting.size, num_classes=dataset_setting.n_classes, batch_size_per_gpu=4,
                            lr=args.lr)


def build_voc_dataset(year: str, root_dir: str, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year=year,
                                                         image_set='train', transforms=train_transforms)
    test_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year=year,
                                                        image_set='trainval', transforms=test_transforms)
    return train_dataset, test_dataset


def build_cityscape_dataset(root_dir: str, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.Cityscapes(root=root_dir, split="train", target_type='semantic',
                                                    transforms=train_transforms)
    test_dataset = torchvision.datasets.Cityscapes(root=root_dir, split="test", target_type='semantic',
                                                   transforms=test_transforms)
    return train_dataset, test_dataset


voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
DATASET_DICT = {
    "voc2012": DataSetSetting(dataset_type="voc2012", size=(256, 256), n_classes=21, label_names=voc_classes,
                              dataset_build_func=lambda root_dir, train_transforms, test_transforms:
                              build_voc_dataset("2012", root_dir, train_transforms, test_transforms)),
    "voc2007": DataSetSetting(dataset_type="voc2007", size=(256, 256), n_classes=21, label_names=voc_classes,
                              dataset_build_func=lambda root_dir, train_transforms, test_transforms:
                              build_voc_dataset("2007", root_dir, train_transforms, test_transforms)),
    "cityscape": DataSetSetting(dataset_type="cityscape", size=(256, 512), n_classes=34,
                                label_names=['ego vehicle', 'rectification', 'out of roi', 'static', 'dynamic',
                                             'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building',
                                             'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole',
                                             'polegroup', 'traffic light', 'traffic sign', 'vegetation',
                                             'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                                             'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',
                                             'license plate'], dataset_build_func=build_cityscape_dataset)
}
MODEL_DICT = {
    "pspnet": build_pspnet,
    "unet": build_unet,
    "custom_shelfnet": build_custom_shelfnet,
    "shallow_shelfnet": build_shallow_shelfnet,
    "shelfnet_realtime": build_shelfnet_realtime,
}


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.Rotate((-30, 30)),
        A.RandomResizedCrop(dataset_setting.size[0], dataset_setting.size[1], scale=(0.5, 2.0)),
        A.CoarseDropout(max_height=int(setting.size[1] / 5), max_width=int(setting.size[0] / 5)),
        A.RandomBrightnessContrast(),
        ToTensorV2(),
    ])
    train_transforms = AlbumentationsSegmentationWrapperTransform(train_transforms, class_num=dataset_setting.n_classes,
                                                                  ignore_indices=[255, ])
    test_transforms = A.Compose([
        A.Resize(dataset_setting.size[0], dataset_setting.size[1]),
        ToTensorV2(),
    ])
    test_transforms = AlbumentationsSegmentationWrapperTransform(test_transforms, class_num=dataset_setting.n_classes,
                                                                 ignore_indices=[255, ])

    train_dataset, test_dataset = setting.dataset_build_func(root_dir, train_transforms, test_transforms)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True), train_dataset, test_dataset


parser = argparse.ArgumentParser(description='Pytorch Image segmentation training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="voc2012", help=f'Dataset type in {list(DATASET_DICT.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="custom_shelfnet", help=f"Model type in {list(MODEL_DICT.keys())}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of sub model(resnet, resnet18, resnet34...).')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DATASET_DICT.get(args.dataset)
    assert dataset_setting is not None, f"Invalid dataset type.  Valid dataset is {list(DATASET_DICT.keys())}"
    img_size = (args.image_size, args.image_size)
    dataset_setting.set_size(img_size)
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)
    # Fetch model and load weight.
    build_model_func = MODEL_DICT.get(args.model)
    assert build_model_func is not None, f"Invalid model type. Valid models is {list(MODEL_DICT.keys())}"
    model: SegmentationModel = try_cuda(build_model_func(dataset_setting, args))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)

    # Training setting.
    lr_scheduler = LearningRateScheduler(max_epoch=args.epoch, base_lr=args.lr) if not isinstance(model,
                                                                                                  ShelfNetRealtime) else None
    callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir="./saved_weights")]
    if args.progress_dir:
        callbacks.append(GenerateSegmentationImageCallback(output_dir=args.progress_dir, per_epoch=1, model=model,
                                                           dataset=test_dataset))
    metric_ls = [SegmentationIoUByClasses(dataset_setting.label_names),
                 SegmentationRecallPrecision(dataset_setting.label_names)]
    metric_for_graph = SegmentationIoUByClasses(dataset_setting.label_names, val_key=MetricKey.KEY_AVERAGE)
    learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=0,
                                                        metric_for_graph=metric_for_graph,
                                                        save_filepath="segmentation_learning_curve.png")

    # Training.
    Trainer(model).fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                       epochs=args.epoch, callbacks=callbacks, lr_scheduler_func=lr_scheduler, metric_ls=metric_ls,
                       calc_metrics_per_epoch=5, learning_curve_visualizer=learning_curve_visualizer)
