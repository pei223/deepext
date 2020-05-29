import argparse
from typing import Tuple
from torchvision.transforms import ToTensor, Resize, Compose, RandomResizedCrop
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext import Trainer, BaseModel, LearningRateScheduler, LabelAndDataTransforms, PSPNet, UNet, ResUNet, ResPSPNet, \
    ModelCheckout, CustomShelfNet, RealtimeSegmentation
from deepext.utils.tensor_util import try_cuda
from deepext.layers import SegmentationAccuracyByClasses, SegmentationIoUByClasses
from deepext.utils import *
from deepext.transforms import ImageToOneHot, PilToTensor, SegmentationLabelSmoothing

from util import DataSetSetting

# TODO モデル・データセットはここを追加
MODEL_PSPNET = "pspnet"
MODEL_UNET = "unet"
MODEL_SHELFNET = "custom_shelfnet"
MODEL_TYPES = [MODEL_PSPNET, MODEL_UNET, MODEL_SHELFNET]
SUBMODEL_RESNET = "resnet"
SUBMODEL_TYPES = [SUBMODEL_RESNET]


def get_model(n_classes, size, model_type: str, submodel_type: str = None):
    # TODO モデルはここを追加
    if MODEL_PSPNET == model_type:
        if submodel_type == "resnet":
            return ResPSPNet(n_classes=n_classes, img_size=size)
        return PSPNet(n_classes=n_classes, img_size=size)
    elif MODEL_UNET == model_type:
        if submodel_type == "resnet":
            return ResUNet(n_input_channels=3, n_output_channels=n_classes)
        return UNet(n_input_channels=3, n_output_channels=n_classes)
    elif MODEL_SHELFNET == model_type:
        return CustomShelfNet(n_classes=n_classes, out_size=size)
    assert f"Invalid model type. Valid models is {MODEL_TYPES}"


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--model', type=str, default=MODEL_PSPNET, help=f"Model type in {MODEL_TYPES}")
parser.add_argument('--load_weight_path', type=str, help="Saved weight path", required=True)
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--n_classes', type=int, help="Class number.", required=True)
parser.add_argument('--submodel', type=str, default=None, help=f'Type of model in {SUBMODEL_TYPES}')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch model and load weight.
    model: SegmentationModel = try_cuda(
        get_model(args.n_classes, args.image_size, model_type=args.model, submodel_type=args.submodel))
    model.load_weight(args.load_weight_path)
    RealtimeSegmentation(model=model, img_size=(args.image_size, args.image_size)).realtime_predict()
