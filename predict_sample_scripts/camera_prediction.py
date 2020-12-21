from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel, DetectionModel, ClassificationModel, AttentionClassificationModel
from deepext.models.segmentation import UNet, ResUNet, CustomShelfNet
from deepext.models.object_detection import EfficientDetector
from deepext.models.classification import EfficientNet, AttentionBranchNetwork, AttentionBranchNetwork, \
    MobileNetV3
from deepext.camera import RealtimeDetection, RealtimeSegmentation, RealtimeAttentionClassification, \
    RealtimeClassification
from deepext.utils import *

# TODO Write path and list
load_weight_path = ""
label_names = []
n_classes = 12

image_size = (512, 512)

# TODO Choose model and load weight.
model = try_cuda(CustomShelfNet(n_classes=n_classes, out_size=image_size, backbone=BackBoneKey.RESNET_18))
model.load_weight(load_weight_path)

if isinstance(model, SegmentationModel):
    RealtimeSegmentation(model=model, img_size_for_model=image_size).realtime_predict(
        video_output_path="output.mp4")
elif isinstance(model, DetectionModel):
    RealtimeDetection(model=model, img_size_for_model=image_size,
                      label_names=label_names).realtime_predict(video_output_path="output.mp4")
elif isinstance(model, AttentionClassificationModel):
    RealtimeAttentionClassification(model=model, img_size_for_model=image_size,
                                    label_names=label_names).realtime_predict(video_output_path="output.mp4")
elif isinstance(model, ClassificationModel):
    RealtimeClassification(model=model, img_size_for_model=image_size,
                           label_names=label_names).realtime_predict(video_output_path="output.mp4")
