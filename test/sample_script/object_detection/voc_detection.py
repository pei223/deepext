from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomRotation, Scale, Compose
import json

from deepext import UNet, PSPNet, ResPSPNet, ResUNet, Trainer, EfficientDetector, ObjectDetectionCollator, \
    VisualizeObjectDetectionResult, M2Det, SSD
from deepext.layers import segmentation_accuracy
from deepext.utils import *

n_classes = 20
img_size = (300, 300)
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

with open("../.env.json") as file:
    settings = json.load(file)

voc_dataset = torchvision.datasets.VOCDetection(root=settings["pascal_voc_2012_detection_root"],
                                                download=False,
                                                transform=Compose([Resize(img_size), ToTensor()]),
                                                target_transform=Compose(
                                                    [VOCAnnotationTransform(class_index_dict, img_size)]))
data_loader = DataLoader(voc_dataset, batch_size=2, shuffle=True, collate_fn=ObjectDetectionCollator())

# model = M2Det(num_classes=n_classes, input_size=320)
model = SSD(num_classes=n_classes, input_size=300)
# model = EfficientDetector(num_classes=n_classes, lr=1e-4, network="efficientdet-d1")

trainer: Trainer = Trainer(model)
trainer.fit(data_loader=data_loader, epochs=epochs,
            lr_scheduler_func=LearningRateScheduler(epochs),
            callbacks=[VisualizeObjectDetectionResult(model, voc_dataset, per_epoch=1, out_dir="../temp")],
            test_dataloader=data_loader, metric_func_ls=[segmentation_accuracy, ])
model.save_weight(".efficientdet.model")
