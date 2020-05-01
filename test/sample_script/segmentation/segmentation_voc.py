from torch.utils.data import DataLoader, Dataset
import random
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomRotation, Scale
import torch
import json

from deepext import UNet, PSPNet, ResPSPNet, ResUNet, Trainer
from deepext.layers import segmentation_accuracy
from deepext.transforms.image_transforms import ImageToOneHot, ReturnFloatTensorToInt
from deepext.utils import *

n_classes = 21
img_size = (256, 256)
epochs = 300

with open("../.env.json") as file:
    settings = json.load(file)

img_transforms = [Resize(img_size), ToTensor(), None, None, ]
label_transforms = [Resize(img_size), ToTensor(), ReturnFloatTensorToInt(), ImageToOneHot(n_classes), ]


def label_img_transform(img, label, img_transforms, label_transforms):
    assert len(img_transforms) == len(label_transforms)
    for i in range(len(img_transforms)):
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        if img_transforms[i] is not None:
            img = img_transforms[i](img)
        if label_transforms[i] is not None:
            label = label_transforms[i](label)
    return img, label


voc_dataset = torchvision.datasets.VOCSegmentation(settings["pascal_voc_2012_root"], year='2012', image_set='train',
                                                   download=False, transform=None, target_transform=None,
                                                   transforms=lambda img, tgt: label_img_transform(img, tgt,
                                                                                                   img_transforms,
                                                                                                   label_transforms))
data_loader = DataLoader(voc_dataset, batch_size=2, shuffle=True)

model: PSPNet = ResPSPNet(n_classes=n_classes, img_size=img_size).cuda()

# model: UNet = ResUNet(n_input_channels=3, n_output_channels=n_classes, loss_type="ce", lr=1e-3).cuda()
# model.load_weight(".unet.model")
trainer: Trainer = Trainer(model)

trainer.fit(data_loader=data_loader, epochs=epochs,
            lr_scheduler_func=LearningRateScheduler(epochs),
            callbacks=[GenerateSegmentationImageCallback(output_dir="../temp", per_epoch=1, model=model,
                                                         dataset=voc_dataset), ],
            test_dataloader=data_loader, metric_func_ls=[segmentation_accuracy, ])
model.save_weight(".unet.model")
# torch.save(model.state_dict(), ".pspnet.model")
