import argparse

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel, DetectionModel, ClassificationModel, AttentionClassificationModel
from deepext.models.segmentation import UNet, ResUNet, CustomShelfNet
from deepext.models.object_detection import EfficientDetector
from deepext.models.classification import EfficientNet, AttentionBranchNetwork, AttentionBranchNetwork, \
    MobileNetV3
from deepext.camera import RealtimeDetection, RealtimeSegmentation, RealtimeAttentionClassification, \
    RealtimeClassification
from deepext.utils import *


def build_unet(args):
    if args.submodel == "resnet":
        return ResUNet(n_input_channels=3, n_output_channels=args.n_classes)
    return UNet(n_input_channels=3, n_output_channels=args.n_classes)


def build_shelfnet(args):
    return CustomShelfNet(n_classes=args.n_classes, out_size=args.image_size,
                          backbone=BackBoneKey.from_val(args.submodel))


def build_efficientdet(args):
    return EfficientDetector(num_classes=args.n_classes, network=f"efficientdet-d{args.model_scale}")


def build_mobilenet(args):
    return MobileNetV3(num_classes=args.n_classes, pretrained=False)


def build_attention_branch_network(args):
    return AttentionBranchNetwork(n_classes=args.n_classes,
                                  backbone=BackBoneKey.from_val(args.submodel))


def build_efficientnet(args):
    return EfficientNet(num_classes=args.n_classes, network=f"efficientnet-d{args.model_scale}")


# NOTE モデル・データセットはここを追加
# TODO ここ他のTrainスクリプトと共通化したい
MODEL_BUILD_DICT = {
    "unet": build_unet,
    "custom_shelfnet": build_shelfnet,
    "efficientdet": build_efficientdet,
    "efficientnet": build_efficientnet,
    "mobilenet": build_mobilenet,
    "attention_branch_network": build_attention_branch_network,
}

parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--model', type=str, required=True, help=f"Model type in {list(MODEL_BUILD_DICT.keys())}")
parser.add_argument('--load_weight_path', type=str, help="Saved weight path", required=True)
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--n_classes', type=int, help="Class number.", required=True)
parser.add_argument('--submodel', type=str, default=None, help=f'Type of model(ResNet, resnet18, resnet34,...).')
parser.add_argument('--model_scale', type=int, default=0, help="Scale of models(EfficientDet, EfficientNet).")
parser.add_argument('--label_names_path', type=str, default="voc_label_names.txt",
                    help="File path of label names (Classification and Detection only)")


def read_label_names(args):
    assert args.label_names_path is not None
    with open(args.label_names_path, "r") as file:
        label_names = []
        for label in file:
            label_names.append(label)
    assert len(label_names) != 0
    return label_names


if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch model and load weight.
    build_model_func = MODEL_BUILD_DICT.get(args.model)
    assert build_model_func is not None, f"Invalid model type:   {args.model}."
    model = try_cuda(build_model_func(args))
    model.load_weight(args.load_weight_path)

    if isinstance(model, SegmentationModel):
        RealtimeSegmentation(model=model, img_size_for_model=(args.image_size, args.image_size)).realtime_predict(
            video_output_path="output.mp4")
    elif isinstance(model, DetectionModel):
        label_names = read_label_names(args)
        RealtimeDetection(model=model, img_size_for_model=(args.image_size, args.image_size),
                          label_names=label_names).realtime_predict(video_output_path="output.mp4")
    elif isinstance(model, AttentionClassificationModel):
        label_names = read_label_names(args)
        RealtimeAttentionClassification(model=model, img_size_for_model=(args.image_size, args.image_size),
                                        label_names=label_names).realtime_predict(video_output_path="output.mp4")
    elif isinstance(model, ClassificationModel):
        label_names = read_label_names(args)
        RealtimeClassification(model=model, img_size_for_model=(args.image_size, args.image_size),
                               label_names=label_names).realtime_predict(video_output_path="output.mp4")
