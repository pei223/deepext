import argparse
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.classification import EfficientNet, MobileNetV3, \
    AttentionBranchNetwork
from deepext.data.transforms import AlbumentationsImageWrapperTransform
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import GenerateAttentionMapCallback, ModelCheckout, CSVClassificationResultCallback
from deepext.metrics.classification import *
from deepext.metrics import MetricKey
from deepext.utils import *

from util import DataSetSetting


# NOTE モデル・データセットはここを追加
def build_efficientnet(dataset_setting, args):
    return EfficientNet(num_classes=dataset_setting.n_classes, lr=args.lr,
                        network=f"efficientnet-b{args.efficientnet_scale}")


def build_mobilenet(dataset_setting, args):
    return MobileNetV3(num_classes=dataset_setting.n_classes, lr=args.lr, pretrained=False)


def build_attention_branch_network(dataset_setting, args):
    return try_cuda(AttentionBranchNetwork(n_classes=dataset_setting.n_classes, lr=args.lr,
                                           backbone=BackBoneKey.from_val(args.submodel)))


def build_stl_dataset(root_dir: str, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="train",
                                               transform=train_transforms)
    test_dataset = torchvision.datasets.STL10(root=root_dir, download=True, split="test",
                                              transform=test_transforms)
    return train_dataset, test_dataset


def build_cifar_dataset(root_dir: str, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=True,
                                                 transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=False,
                                                transform=test_transforms)
    return train_dataset, test_dataset


DATASET_DICT = {
    "stl": DataSetSetting(dataset_type="stl", size=(96, 96), n_classes=10,
                          label_names=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship',
                                       'truck'], dataset_build_func=build_stl_dataset),
    "cifar": DataSetSetting(dataset_type="cifar", size=(32, 32), n_classes=10,
                            label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                                         'ship', 'truck'], dataset_build_func=build_cifar_dataset)
}
MODEL_DICT = {
    "efficientnet": build_efficientnet,
    "mobilenet": build_mobilenet,
    "attention_branch_network": build_attention_branch_network
}


def get_dataloader(setting: DataSetSetting, root_dir: str, batch_size: int) -> Tuple[
    DataLoader, DataLoader, Dataset, Dataset]:
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.RandomResizedCrop(setting.size[0], setting.size[1], scale=(0.5, 2.0)),
        A.Rotate((-30, 30)),
        ToTensorV2(),
    ])
    train_transforms = AlbumentationsImageWrapperTransform(train_transforms)

    test_transforms = A.Compose([
        A.Resize(setting.size[0], setting.size[1]),
        ToTensorV2(),
    ])
    test_transforms = AlbumentationsImageWrapperTransform(test_transforms)
    train_dataset, test_dataset = setting.dataset_build_func(root_dir, train_transforms, test_transforms)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True), train_dataset, test_dataset


parser = argparse.ArgumentParser(description='Pytorch Image classification training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="stl", help=f'Dataset type in {list(DATASET_DICT.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="mobilenet", help=f"Model type in {list(MODEL_DICT.keys())}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientnet_scale', type=int, default=0, help="Number of scale of EfficientNet.")
parser.add_argument('--image_size', type=int, default=None, help="Image size(default is 256)")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of submodel(resnet18, resnet34...).')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_setting = DATASET_DICT.get(args.dataset)
    assert dataset_setting is not None, f"Invalid dataset type.  Valid dataset is {list(DATASET_DICT.keys())}"
    if args.image_size:
        img_size = (args.image_size, args.image_size)
        dataset_setting.set_size(img_size)

    # Fetch model and load weight.
    build_model_func = MODEL_DICT.get(args.model)
    assert build_model_func is not None, f"Invalid model type. Valid models is {list(MODEL_DICT.keys())}"
    model = try_cuda(build_model_func(dataset_setting, args))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)

    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(dataset_setting, args.dataset_root,
                                                                                    args.batch_size)
    # Training setting.
    lr_scheduler = CosineDecayScheduler(max_lr=args.lr, max_epochs=args.epoch, warmup_epochs=0)
    callbacks = [ModelCheckout(per_epoch=int(args.epoch / 5), model=model, our_dir="saved_weights"),
                 CSVClassificationResultCallback(model=model, per_epoch=args.epoch, dataset=test_dataset,
                                                 label_names=dataset_setting.label_names,
                                                 out_filepath=f"{args.progress_dir}/result.csv")]
    if args.progress_dir:
        if isinstance(model, AttentionBranchNetwork):
            callbacks.append(GenerateAttentionMapCallback(model=model, output_dir=args.progress_dir, per_epoch=5,
                                                          dataset=test_dataset,
                                                          label_names=dataset_setting.label_names))
    metric_ls = [ClassificationAccuracyByClasses(dataset_setting.label_names), ]
    metric_for_graph = ClassificationAccuracyByClasses(dataset_setting.label_names, val_key=MetricKey.KEY_TOTAL)
    learning_curve_visualizer = LearningCurveVisualizer(metric_name="Accuracy", ignore_epoch=0,
                                                        metric_for_graph=metric_for_graph,
                                                        save_filepath="classification_learning_curve.png")
    # Training.
    Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(data_loader=train_dataloader,
                                                                            test_dataloader=test_dataloader,
                                                                            epochs=args.epoch, callbacks=callbacks,
                                                                            lr_scheduler_func=lr_scheduler,
                                                                            metric_ls=metric_ls,
                                                                            calc_metrics_per_epoch=5)
