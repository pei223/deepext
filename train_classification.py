import argparse
from torchvision.transforms import Resize, Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, \
    RandomRotation, RandomErasing
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext.base import BaseModel
from deepext.classifier import AttentionBranchNetwork, EfficientNet, MobileNetV3
from deepext.trainer import Trainer
from deepext.trainer.callbacks import GenerateAttentionMapCallback, LearningRateScheduler, ModelCheckout
from deepext.metrics import ClassificationAccuracyByClasses
from deepext.utils import *

from util import DataSetSetting

# NOTE モデル・データセットはここを追加
MODEL_EFFICIENT_NET = "efficientnet"
MODEL_ATTENTION_BRANCH_NETWORK = "attention_branch_network"
MODEL_MOBILENET = "mobilenet"
MODEL_TYPES = [MODEL_EFFICIENT_NET, MODEL_ATTENTION_BRANCH_NETWORK, MODEL_MOBILENET]
DATASET_STL = "stl"
DATASET_CIFAR = "cifar"
DATASET_TYPES = [DATASET_STL, DATASET_CIFAR]
settings = [DataSetSetting(dataset_type=DATASET_STL, size=(96, 96), n_classes=10,
                           label_names=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship',
                                        'truck']),
            DataSetSetting(dataset_type=DATASET_CIFAR, size=(32, 32), n_classes=10,
                           label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                        'truck'])]


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = Compose(
        [RandomHorizontalFlip(),
         ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
         RandomResizedCrop(size=setting.size, scale=(0.5, 2.0)),
         RandomRotation((-10, 10)),
         ToTensor(),
         RandomErasing(), ])
    test_transforms = Compose([Resize(setting.size), ToTensor()])
    train_dataset, test_dataset = None, None
    # NOTE データセットはここを追加
    if DATASET_STL == setting.dataset_type:
        train_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="train",
                                                   transform=train_transforms)
        test_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="test",
                                                  transform=test_transforms)
    elif DATASET_CIFAR == setting.dataset_type:
        train_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=True,
                                                     transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=False,
                                                    transform=test_transforms)
    assert train_dataset is not None and test_dataset is not None, f"Not supported setting: {setting.dataset_type}"
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True), train_dataset, test_dataset


def get_model(dataset_setting: DataSetSetting, model_type: str, lr: float, efficientnet_scale: int = 0):
    # NOTE モデルはここを追加
    if MODEL_EFFICIENT_NET == model_type:
        return EfficientNet(num_classes=dataset_setting.n_classes, lr=lr, network=f"efficientnet-b{efficientnet_scale}")
    elif MODEL_ATTENTION_BRANCH_NETWORK == model_type:
        return try_cuda(AttentionBranchNetwork(n_classes=dataset_setting.n_classes, lr=lr))
    elif MODEL_MOBILENET == model_type:
        return MobileNetV3(num_classes=dataset_setting.n_classes, lr=lr, pretrained=False)
    assert f"Invalid model type. Valid models is {MODEL_TYPES}"


parser = argparse.ArgumentParser(description='Pytorch Image classification training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default=DATASET_STL, help=f'Dataset type in {DATASET_TYPES}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default=MODEL_MOBILENET, help=f"Model type in {MODEL_TYPES}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientnet_scale', type=int, default=0, help="Number of scale of EfficientNet.")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DataSetSetting.from_dataset_type(settings, args.dataset)
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)
    # Fetch model and load weight.
    model: BaseModel = try_cuda(
        get_model(dataset_setting, model_type=args.model, lr=args.lr, efficientnet_scale=args.efficientnet_scale))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)
    save_weight_path = args.save_weight_path or f"./{args.model}.pth"

    # Training.
    callbacks = [ModelCheckout(per_epoch=10, model=model, our_dir="./saved_weights")]
    if args.progress_dir:
        if isinstance(model, AttentionBranchNetwork):
            callbacks = [
                GenerateAttentionMapCallback(model=model, output_dir=args.progress_dir, per_epoch=1,
                                             dataset=test_dataset)]

    trainer = Trainer(model)
    trainer.fit(data_loader=train_dataloader, test_dataloader=test_dataloader,
                epochs=args.epoch, callbacks=callbacks,
                lr_scheduler_func=LearningRateScheduler(args.epoch),
                metric_ls=[ClassificationAccuracyByClasses(dataset_setting.label_names), ])

    # Save weight.
    model.save_weight(save_weight_path)
