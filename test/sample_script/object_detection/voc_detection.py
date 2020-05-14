from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomRotation, Scale, Compose, Normalize
import json

from deepext import UNet, PSPNet, ResPSPNet, ResUNet, Trainer, EfficientDetector, ObjectDetectionCollator, \
    VisualizeRandomObjectDetectionResult, M2Det, SSD
from deepext.layers import segmentation_accuracy
from deepext.utils import *

n_classes = 20
img_size = (512, 512)
epochs = 300

class_index_dict = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}
label_names = list(class_index_dict.keys())

with open("../.env.json") as file:
    settings = json.load(file)

voc_dataset = torchvision.datasets.VOCDetection(root=settings["pascal_voc_2007_detection_root"],
                                                download=True,
                                                year='2007',
                                                transform=Compose([Resize(img_size), ToTensor(),
                                                                   # Normalize(mean=(0.485, 0.456, 0.406),
                                                                   #           std=(0.229, 0.224, 0.225))
                                                                   ]),
                                                target_transform=Compose(
                                                    [VOCAnnotationTransform(class_index_dict, img_size)]))
# voc_dataset = Subset(voc_dataset, [i for i in range(100)])

data_loader = DataLoader(voc_dataset, batch_size=8, shuffle=True, collate_fn=ObjectDetectionCollator())

# model = M2Det(num_classes=n_classes, input_size=320)
# model = SSD(num_classes=n_classes, input_size=300)
model = EfficientDetector(num_classes=n_classes, lr=1e-4, network="efficientdet-d0",
                          backbone_path="C:/Users/ip-miyake/Downloads/efficientnet-b0-355c32eb.pth")
model.load_weight("C:/Users/ip-miyake/Downloads/checkpoint_VOC_efficientdet-d0_268.pth")

trainer: Trainer = Trainer(model)
trainer.fit(data_loader=data_loader, epochs=epochs,
            lr_scheduler_func=LearningRateScheduler(epochs, power=1),
            callbacks=[
                VisualizeRandomObjectDetectionResult(model, img_size, voc_dataset, per_epoch=1, out_dir="../temp",
                                                     label_names=label_names)],
            test_dataloader=data_loader, metric_func_ls=[])
model.save_weight(".efficientdet.model")
