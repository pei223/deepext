import argparse
from deepext import BaseModel, PSPNet, UNet, ResUNet, ResPSPNet, CustomShelfNet, EfficientDetector, SegmentationModel, \
    DetectionModel, RealtimeDetection, RealtimeSegmentation
from deepext.utils.tensor_util import try_cuda
from deepext.utils import *

# TODO モデル・データセットはここを追加
MODEL_PSPNET = "pspnet"
MODEL_UNET = "unet"
MODEL_SHELFNET = "custom_shelfnet"
MODEL_EFFICIENTDET = "efficientdet"
MODEL_TYPES = [MODEL_PSPNET, MODEL_UNET, MODEL_SHELFNET, MODEL_EFFICIENTDET]
SUBMODEL_RESNET = "resnet"
SUBMODEL_TYPES = [SUBMODEL_RESNET]


def get_model(n_classes, size, model_type: str, submodel_type: str = None, model_scale: int = 0):
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
    elif MODEL_EFFICIENTDET == model_type:
        return EfficientDetector(num_classes=n_classes, network=f"efficientdet-d{model_scale}")
    assert f"Invalid model type. Valid models is {MODEL_TYPES}"


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--model', type=str, default=MODEL_PSPNET, help=f"Model type in {MODEL_TYPES}")
parser.add_argument('--load_weight_path', type=str, help="Saved weight path", required=True)
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--n_classes', type=int, help="Class number.", required=True)
parser.add_argument('--submodel', type=str, default=None, help=f'Type of model in {SUBMODEL_TYPES}')
parser.add_argument('--efficientdet_scale', type=int, default=0, help="Scale of EfficientDet.")
parser.add_argument('--label_names_path', type=str, default="voc_label_names.txt",
                    help="File path of label names (Detection only)")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch model and load weight.
    model = try_cuda(
        get_model(args.n_classes, args.image_size, model_type=args.model, submodel_type=args.submodel,
                  model_scale=args.efficientdet_scale))
    model.load_weight(args.load_weight_path)

    if isinstance(model, SegmentationModel):
        RealtimeSegmentation(model=model, img_size=(args.image_size, args.image_size)).realtime_predict()
    elif isinstance(model, DetectionModel):
        assert args.label_names_path is not None
        with open(args.label_names_path, "r") as file:
            label_names = []
            for label in file:
                label_names.append(label)
        assert len(label_names) != 0
        RealtimeDetection(model=model, img_size=(args.image_size, args.image_size),
                          label_names=label_names).realtime_predict()
