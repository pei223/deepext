from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomResizedCrop, Compose
import json

from deepext import singan
from deepext.data.dataset.gan import SimpleGANDataSet

from deepext.utils import *

img_size = (256, 256)
epochs = 100000

with open("../.env.json") as file:
    settings = json.load(file)

img_transforms = Compose(
    [Resize(img_size), RandomHorizontalFlip(), RandomResizedCrop(size=img_size, scale=(0.75, 1.25)), ToTensor()])

dataset = SimpleGANDataSet(root_dir=settings["singan_root"], transform=img_transforms)

data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

scale_list = [24, 32, 64, 96, 128, 160, 192, 224, 256]

G_list = [singan.Generator(in_channels=3, middle_channels=int(32 * (1 + i // 3))).cuda() for i in
          range(len(scale_list))]
D_list = [singan.Discriminator(in_channels=3, middle_channels=int(32 * (1 + i // 3))).cuda() for i in
          range(len(scale_list))]

singan_model = singan.SinGAN(G_list, D_list)
singan_model.fit(data_loader=data_loader, epochs=epochs, size_list=scale_list, lr=1e-4,
                 on_epoch_finished=[
                     singan.SuperResolutionCallback(dataset=dataset, singan=singan_model,
                                                    base_img_path=settings["singan_test_image"],
                                                    out_dir="../temp", per_epoch=1000),
                     singan.RandomRealisticImageCallback(dataset=dataset, singan=singan_model,
                                                         base_img_path=settings["singan_test_image"],
                                                         out_dir="../temp", per_epoch=1000,
                                                         size_list=scale_list)
                 ])
