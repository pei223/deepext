from typing import Tuple

from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomRotation, Compose, RandomResizedCrop
import json
from deepext import AttentionBranchNetwork, Trainer, random_subset_dataset
from deepext.layers import classification_accuracy

from deepext.utils import *

with open("../.env.json") as file:
    settings = json.load(file)


def stl10_setting(preset_transforms: List) -> Tuple[DataLoader, Dataset, int, DataLoader]:
    size = (96, 96)
    img_transforms = Compose([Resize(size), RandomResizedCrop(size=size, scale=(0.1, 1.0))] + preset_transforms)
    dataset = torchvision.datasets.STL10(root=settings["stl10_root"], download=True, split="train",
                                         transform=img_transforms)

    test_dataset = torchvision.datasets.STL10(root=settings["stl10_root"], download=True, split="test",
                                              transform=Compose([Resize(size), ToTensor()]))
    return DataLoader(dataset, batch_size=8, shuffle=True), dataset, 10, DataLoader(test_dataset, batch_size=8,
                                                                                    shuffle=True)


def cifar10_setting(preset_transforms: List) -> Tuple[DataLoader, Dataset, int, DataLoader]:
    size = (32, 32)
    img_transforms = Compose([Resize(size), RandomResizedCrop(size=size, scale=(0.1, 1.0)), ] + preset_transforms)
    dataset = torchvision.datasets.CIFAR10(root=settings["cifar10_root"], download=True, train=True,
                                           transform=img_transforms)
    test_dataset = torchvision.datasets.CIFAR10(root=settings["cifar10_root"], download=True, train=False,
                                                transform=Compose([Resize(size), ToTensor()]))
    return DataLoader(dataset, batch_size=64, shuffle=True), dataset, 10, DataLoader(test_dataset, batch_size=8,
                                                                                     shuffle=True)


dataset_type = "stl10"
epochs = 300
preset_transforms = [RandomHorizontalFlip(), RandomRotation(180), ToTensor(), ]

if dataset_type == "stl10":
    data_loader, dataset, n_classes, test_data_loader = stl10_setting(preset_transforms)
elif dataset_type == "cifar10":
    data_loader, dataset, n_classes, test_data_loader = cifar10_setting(preset_transforms)

model: AttentionBranchNetwork = AttentionBranchNetwork(n_classes=n_classes, first_layer_channels=32, lr=1e-4).cuda()
# model.load_weight(".abn.model")
trainer: Trainer = Trainer(model)
trainer.fit(data_loader, epochs=epochs, lr_scheduler_func=LearningRateScheduler(epochs),
            callbacks=[GenerateAttentionMapCallback(model=model, output_dir="../temp", per_epoch=1, dataset=dataset)],
            test_dataloader=test_data_loader, metric_func_ls=[classification_accuracy, ])
model.save_weight(".abn.model")
