from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.data.dataset import VOCDataset, AdjustDetectionTensorCollator
from deepext.data.transforms import AlbumentationsDetectionWrapperTransform
from deepext.layers.subnetwork import *

image_dir = "D:/dataset/object_detection/pfood/images"
annotation_dir = "D:/dataset/object_detection/pfood/annotation"
label_names = ["food", "drink"]
class_index_dict = {}
for i in range(len(label_names)):
    class_index_dict[label_names[i]] = i

test_transforms = AlbumentationsDetectionWrapperTransform([
    A.Resize(512, 512),
    ToTensorV2(),
], )
test_dataset = VOCDataset(class_index_dict=class_index_dict, image_dir_path=image_dir,
                          annotation_dir_path=annotation_dir, transforms=test_transforms)

test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True,
                             collate_fn=AdjustDetectionTensorCollator())

en = EfficientNetBackBone(model_type=BackBoneKey.EFFICIENTNET_B0)
for train_x, teacher in test_dataloader:
    result = en(train_x)
    for r in result:
        print(r.shape)
    break
