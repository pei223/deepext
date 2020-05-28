import argparse
from typing import Tuple
from torchvision.transforms import ToTensor, Resize, Compose, RandomResizedCrop
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext import Trainer, BaseModel, LearningRateScheduler, LabelAndDataTransforms, PSPNet, UNet, ResUNet, ResPSPNet, \
    ModelCheckout, ShelfNet
from deepext.utils.tensor_util import try_cuda
from deepext.layers import SegmentationAccuracyByClasses, SegmentationIoUByClasses
from deepext.utils import *
from deepext.transforms import ImageToOneHot, PilToTensor, SegmentationLabelSmoothing

from util import DataSetSetting

# TODO モデル・データセットはここを追加
MODEL_PSPNET = "pspnet"
MODEL_UNET = "unet"
MODEL_SHELFNET = "shelfnet"
MODEL_TYPES = [MODEL_PSPNET, MODEL_UNET, MODEL_SHELFNET]
SUBMODEL_RESNET = "resnet"
SUBMODEL_TYPES = [SUBMODEL_RESNET]
DATASET_VOC2012 = "voc2012"
DATASET_VOC2007 = "voc2007"
DATASET_TYPES = [DATASET_VOC2007, DATASET_VOC2012]
settings = [DataSetSetting(dataset_type=DATASET_VOC2012, size=(256, 256), n_classes=21,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"]),
            DataSetSetting(dataset_type=DATASET_VOC2007, size=(256, 256), n_classes=21,
                           label_names=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"])]


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = LabelAndDataTransforms([
        (Resize(setting.size), Resize(setting.size)), (ToTensor(), PilToTensor()),
        (None, ImageToOneHot(setting.n_classes)), (None, SegmentationLabelSmoothing(setting.n_classes))
    ])
    test_transforms = LabelAndDataTransforms([
        (Resize(setting.size), Resize(setting.size)), (ToTensor(), PilToTensor()),
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
    elif MODEL_SHELFNET == model_type:
        return ShelfNet(n_classes=dataset_setting.n_classes, lr=lr, out_size=dataset_setting.size)
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
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of model in {SUBMODEL_TYPES}')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DataSetSetting.from_dataset_type(settings, args.dataset)
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
                lr_scheduler_func=LearningRateScheduler(args.epoch),
                metric_func_ls=[SegmentationIoUByClasses(dataset_setting.label_names)])

    # Save weight.
    model.save_weight(save_weight_path)
