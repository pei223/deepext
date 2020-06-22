import argparse
from torchvision.transforms import Resize, RandomResizedCrop, RandomHorizontalFlip, RandomErasing, \
    ColorJitter, RandomRotation
import torchvision
from torch.utils.data import DataLoader

from deepext import Trainer, PSPNet, UNet, ResUNet, ResPSPNet, \
    CustomShelfNet, ShelfNetRealtime, LearningCurveVisualizer
from deepext.data.transforms.image_transforms import LabelAndDataTransforms
from deepext.metrics import SegmentationAccuracyByClasses, SegmentationIoUByClasses, MetricKey
from deepext.utils import *
from deepext.data.transforms import ImageToOneHot, PilToTensor
from util import DataSetSetting

# TODO モデル・データセットはここを追加
MODEL_PSPNET = "pspnet"
MODEL_UNET = "unet"
MODEL_SHELFNET_REALTIME = "shelfnet_realtime"
MODEL_CUSTOM_SHELFNET = "custom_shelfnet"
MODEL_TYPES = [MODEL_PSPNET, MODEL_UNET, MODEL_CUSTOM_SHELFNET, MODEL_SHELFNET_REALTIME]
SUBMODEL_RESNET = "resnet"
SUBMODEL_TYPES = [SUBMODEL_RESNET]
DATASET_VOC2012 = "voc2012"
DATASET_VOC2007 = "voc2007"
DATASET_CITYSCAPE = "cityscape"
DATASET_TYPES = [DATASET_VOC2007, DATASET_VOC2012, DATASET_CITYSCAPE]
settings = [DataSetSetting(dataset_type=DATASET_VOC2012, size=(256, 256), n_classes=21,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"]),
            DataSetSetting(dataset_type=DATASET_VOC2007, size=(256, 256), n_classes=21,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"]),
            DataSetSetting(dataset_type=DATASET_CITYSCAPE, size=(256, 512), n_classes=34,
                           label_names=[
                               'ego vehicle', 'rectification', 'out of roi', 'static', 'dynamic', 'ground', 'road',
                               'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                               'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                               'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train',
                               'motorcycle', 'bicycle', 'license plate'
                           ])]


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = LabelAndDataTransforms([
        (RandomHorizontalFlip(), RandomHorizontalFlip()),
        (RandomResizedCrop(size=dataset_setting.size, scale=(0.5, 2.0)),
         RandomResizedCrop(size=dataset_setting.size, scale=(0.5, 2.0))),
        (RandomRotation((-10, 10)), RandomRotation((-10, 10))),
        (ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), None),
        (ToTensor(), PilToTensor()),
        (RandomErasing(), RandomErasing()),
        (None, ImageToOneHot(setting.n_classes)),
    ])
    test_transforms = LabelAndDataTransforms([
        (Resize(setting.size), Resize(setting.size)),
        (ToTensor(), PilToTensor()),
        (None, ImageToOneHot(setting.n_classes))
    ])
    train_dataset, test_dataset = None, None
    class_index_dict = {}
    for i, label_name in enumerate(setting.label_names):
        class_index_dict[label_name] = i
    # TODO データセットはここを追加
    if DATASET_VOC2012 == setting.dataset_type:
        train_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year='2012',
                                                             image_set='train', transforms=train_transforms)
        test_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year='2012',
                                                            image_set='trainval', transforms=test_transforms)
    elif DATASET_VOC2007 == setting.dataset_type:
        train_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year='2007',
                                                             image_set='train', transforms=train_transforms)
        test_dataset = torchvision.datasets.VOCSegmentation(root=root_dir, download=True, year='2007',
                                                            image_set='trainval', transforms=test_transforms)
    elif DATASET_CITYSCAPE == setting.dataset_type:
        train_dataset = torchvision.datasets.Cityscapes(root=root_dir, split="train", target_type='semantic',
                                                        transforms=train_transforms)
        test_dataset = torchvision.datasets.Cityscapes(root=root_dir, split="test", target_type='semantic',
                                                       transforms=test_transforms)
    assert train_dataset is not None and test_dataset is not None, f"Not supported setting: {setting.dataset_type}"
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True), train_dataset, test_dataset


def get_model(dataset_setting: DataSetSetting, model_type: str, lr: float, submodel_type: str = None):
    # TODO モデルはここを追加
    if MODEL_PSPNET == model_type:
        if submodel_type == "resnet":
            return ResPSPNet(n_classes=dataset_setting.n_classes, img_size=dataset_setting.size, lr=lr)
        return PSPNet(n_classes=dataset_setting.n_classes, img_size=dataset_setting.size, lr=lr)
    elif MODEL_UNET == model_type:
        if submodel_type == "resnet":
            return ResUNet(n_input_channels=3, n_output_channels=dataset_setting.n_classes, lr=lr)
        return UNet(n_input_channels=3, n_output_channels=dataset_setting.n_classes, lr=lr)
    elif MODEL_CUSTOM_SHELFNET == model_type:
        return CustomShelfNet(n_classes=dataset_setting.n_classes, lr=lr, out_size=dataset_setting.size,
                              loss_type="ce", backbone=submodel_type)
    elif MODEL_SHELFNET_REALTIME == model_type:
        return ShelfNetRealtime(size=dataset_setting.size, num_classes=dataset_setting.n_classes, batch_size_per_gpu=4,
                                lr=lr)
    assert f"Invalid model type. Valid models is {MODEL_TYPES}"


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default=DATASET_VOC2012, help=f'Dataset type in {DATASET_TYPES}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default=MODEL_PSPNET, help=f"Model type in {MODEL_TYPES}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--image_size', type=int, default=None, help="Image size(default is 256)")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of model in {SUBMODEL_TYPES}')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DataSetSetting.from_dataset_type(settings, args.dataset)
    if args.image_size:
        img_size = (args.image_size, args.image_size)
        dataset_setting.set_size(img_size)
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)
    # Fetch model and load weight.
    model: SegmentationModel = try_cuda(
        get_model(dataset_setting, model_type=args.model, lr=args.lr, submodel_type=args.submodel))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)
    save_weight_path = args.save_weight_path or f"./{args.model}.pth"

    # Training.
    callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir="./saved_weights")]
    if args.progress_dir:
        callbacks.append(GenerateSegmentationImageCallback(output_dir=args.progress_dir, per_epoch=1, model=model,
                                                           dataset=test_dataset))

    trainer = Trainer(model)
    trainer.fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                epochs=args.epoch, callbacks=callbacks,
                lr_scheduler_func=LearningRateScheduler(args.epoch) if not isinstance(model,
                                                                                      ShelfNetRealtime) else None,
                metric_ls=[SegmentationAccuracyByClasses(dataset_setting.label_names),
                           SegmentationIoUByClasses(dataset_setting.label_names)],
                calc_metrics_per_epoch=5,
                learning_curve_visualizer=LearningCurveVisualizer(metric_name="mIoU",
                                                                  ignore_epoch=0,
                                                                  metric_for_graph=SegmentationIoUByClasses(
                                                                      dataset_setting.label_names,
                                                                      val_key=MetricKey.KEY_AVERAGE),
                                                                  save_filepath="learning_curve.png"))
    # Save weight.
    model.save_weight(save_weight_path)
