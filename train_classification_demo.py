import argparse
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.utils.data import DataLoader, Dataset

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import ClassificationModel
from deepext.models.classification import *
from deepext.data.transforms import AlbumentationsOnlyImageWrapperTransform
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import GenerateAttentionMapCallback, ModelCheckout, CSVClassificationResultCallback
from deepext.metrics.classification import *
from deepext.metrics import DetailMetricKey
from deepext.utils import *

from dataset_info import CLASSIFICATION_DATASET_INFO

VALID_MODEL_KEYS = ["efficientnet", "mobilenet", "abn", "custommodel"]


# NOTE モデル・データセットはここを追加
def build_model(args, n_classes) -> ClassificationModel:
    if args.model == "efficientnet":
        return EfficientNet(num_classes=n_classes, lr=args.lr, network=f"efficientnet-b{args.efficientnet_scale}")
    if args.model == "mobilenet":
        return MobileNetV3(num_classes=n_classes, lr=args.lr, pretrained=False)
    if args.model == "abn":
        return AttentionBranchNetwork(n_classes=n_classes, lr=args.lr, backbone=BackBoneKey.from_val(args.submodel))
    if args.model == "custommodel":
        return CustomClassificationNetwork(n_classes=n_classes, lr=args.lr,
                                           backbone=BackBoneKey.from_val(args.submodel))
    raise RuntimeError(f"Invalid model name: {args.model}")


def build_transforms(args) -> Tuple[any, any]:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomResizedCrop(width=args.image_size, height=args.image_size, scale=(0.7, 1.2)),
        A.Rotate((-30, 30), p=0.3),
        A.CoarseDropout(max_width=int(args.image_size / 8), max_height=int(args.image_size / 8), max_holes=3, p=0.3),
        ToTensorV2(),
    ])
    train_transforms = AlbumentationsOnlyImageWrapperTransform(train_transforms)

    test_transforms = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
        ToTensorV2(),
    ])
    test_transforms = AlbumentationsOnlyImageWrapperTransform(test_transforms)
    return train_transforms, test_transforms


def build_dataset(args, train_transforms, test_transforms) -> Tuple[Dataset, Dataset]:
    if args.dataset == "stl10":
        train_dataset = torchvision.datasets.STL10(root=args.dataset_root, download=True, split="train",
                                                   transform=train_transforms)
        test_dataset = torchvision.datasets.STL10(root=args.dataset_root, download=True, split="test",
                                                  transform=test_transforms)
        return train_dataset, test_dataset
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, download=True, train=True,
                                                     transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, download=True, train=False,
                                                    transform=test_transforms)
        return train_dataset, test_dataset
    raise RuntimeError(f"Invalid dataset name: {args.dataset_root}")


def build_data_loader(args, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


parser = argparse.ArgumentParser(description='Pytorch Image classification training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="stl10",
                    help=f'Dataset type in {list(CLASSIFICATION_DATASET_INFO.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="mobilenet", help=f"Model type in {VALID_MODEL_KEYS}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientnet_scale', type=int, default=0, help="Number of scale of EfficientNet.")
parser.add_argument('--image_size', type=int, default=96, help="Image size.")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of submodel(resnet18, resnet34...).')

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_info = CLASSIFICATION_DATASET_INFO.get(args.dataset)
    if dataset_info is None:
        raise ValueError(
            f"Invalid dataset name - {args.dataset}.  Required [{list(CLASSIFICATION_DATASET_INFO.keys())}]")

    label_names = dataset_info["label_names"]
    class_index_dict = {}
    for i, label_name in enumerate(label_names):
        class_index_dict[label_name] = i

    # Fetch dataset.
    train_transforms, test_transforms = build_transforms(args)
    train_dataset, test_dataset = build_dataset(args, train_transforms, test_transforms)
    train_data_loader, test_data_loader = build_data_loader(args, train_dataset, test_dataset)

    # Fetch model and load weight.
    model = try_cuda(build_model(args, dataset_info["n_classes"]))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)

    # Training setting.
    loss_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.get_optimizer(), patience=3, verbose=True)
    ep_lr_scheduler = None
    # loss_lr_scheduler = None
    # ep_lr_scheduler = CosineDecayScheduler(max_lr=args.lr, max_epochs=args.epoch, warmup_epochs=0)
    callbacks = [ModelCheckout(per_epoch=int(args.epoch / 5), model=model, our_dir="saved_weights"),
                 CSVClassificationResultCallback(model=model, per_epoch=args.epoch, dataset=test_dataset,
                                                 label_names=label_names,
                                                 out_filepath=f"{args.progress_dir}/result.csv")]
    if args.progress_dir:
        if isinstance(model, AttentionBranchNetwork):
            callbacks.append(GenerateAttentionMapCallback(model=model, output_dir=args.progress_dir, per_epoch=5,
                                                          dataset=test_dataset,
                                                          label_names=label_names))
    metric_ls = [ClassificationAccuracyByClasses(label_names), ]
    metric_for_graph = ClassificationAccuracyByClasses(label_names, val_key=DetailMetricKey.KEY_TOTAL)
    learning_curve_visualizer = LearningCurveVisualizer(metric_name="Accuracy", ignore_epoch=0,
                                                        save_filepath="classification_learning_curve.png")
    # Training.
    Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(train_data_loader=train_data_loader,
                                                                            test_data_loader=test_data_loader,
                                                                            epochs=args.epoch, callbacks=callbacks,
                                                                            epoch_lr_scheduler_func=ep_lr_scheduler,
                                                                            loss_lr_scheduler=loss_lr_scheduler,
                                                                            metric_for_graph=metric_for_graph,
                                                                            metric_ls=metric_ls,
                                                                            calc_metrics_per_epoch=5)
