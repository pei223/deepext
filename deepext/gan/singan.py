from typing import Tuple, Callable
from torch import nn
from torch.nn import functional as F
import torch
from torchvision.transforms import ToTensor, ToPILImage

from torch.utils.data import DataLoader
from statistics import mean

from ..layers import *
from ..utils import *


class Generator(nn.Module):
    def __init__(self, in_channels: int, middle_channels=64):
        super(Generator, self).__init__()
        self._model = nn.Sequential(
            Conv2DBatchNormLeakyRelu(in_channels=in_channels, out_channels=middle_channels, ),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=middle_channels, ),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=middle_channels, ),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=middle_channels, ),
            nn.Conv2d(in_channels=middle_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor or None, pre_output: torch.Tensor or None):
        if pre_output is not None:
            return self._model(noise + pre_output) + pre_output
        return self._model(noise)

    def forward_image(self, image):
        return self._model(image) + image


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, middle_channels=64):
        super(Discriminator, self).__init__()
        self._model = nn.Sequential(
            Conv2DBatchNormLeakyRelu(in_channels=in_channels, out_channels=middle_channels),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=middle_channels),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=middle_channels),
            Conv2DBatchNormLeakyRelu(in_channels=middle_channels, out_channels=2),
            GlobalAveragePooling(),
        )

    def forward(self, x):
        output = self._model(x)
        return F.softmax(output.view(output.shape[0], -1), dim=1)


class SinGAN:
    def __init__(self, generators: List[Generator], discriminators: List[Discriminator]):
        self.G_list = generators
        self.D_list = discriminators

    def fit(self, data_loader: DataLoader, epochs: int, on_epoch_finished: List[Callable[[int], None]] or None = None,
            lr=1e-3, size_list=None):
        G_optimizer_ls = [torch.optim.Adam(self.G_list[i].parameters(), lr=lr) for i in range(len(size_list))]
        D_optimizer_ls = [torch.optim.Adam(self.D_list[i].parameters(), lr=lr) for i in range(len(size_list))]
        for epoch in range(epochs):
            self._to_train_models(self.G_list)
            self._to_train_models(self.D_list)
            G_mean_loss, D_mean_loss, reconstruction_mean_loss = self.train_step(data_loader, G_optimizer_ls,
                                                                                 D_optimizer_ls, size_list)
            print(
                f"epoch {epoch + 1} / {epochs} --- G loss: {G_mean_loss}, D loss: {D_mean_loss}, rec loss: {reconstruction_mean_loss}")
            self._to_eval_models(self.G_list)
            self._to_eval_models(self.D_list)
            if on_epoch_finished:
                for callback in on_epoch_finished:
                    callback(epoch)

    def train_step(self, data_loader: DataLoader, G_optimizer_ls, D_optimizer_ls, size_list) -> Tuple[
        float, float, float]:
        G_loss_list, D_loss_list, reconstruction_loss_list = [], [], []
        for train_x in data_loader:
            train_x = try_cuda(train_x)
            pre_output = None
            for i, size in enumerate(size_list):
                real_image = F.interpolate(train_x, (size, size), mode="bilinear")
                pre_output = F.interpolate(pre_output, (size, size),
                                           mode="bilinear") if pre_output is not None else None
                pre_output, G_loss, D_loss, reconstruction_loss = self.train_one_scale(generator=self.G_list[i],
                                                                                       discriminator=self.D_list[i],
                                                                                       G_optimizer=G_optimizer_ls[i],
                                                                                       D_optimizer=D_optimizer_ls[i],
                                                                                       real_image=real_image,
                                                                                       pre_output=pre_output,
                                                                                       noise_size=(
                                                                                           data_loader.batch_size,
                                                                                           train_x.shape[1], size,
                                                                                           size))
                G_loss_list.append(G_loss)
                D_loss_list.append(D_loss)
                reconstruction_loss_list.append(reconstruction_loss)
            return mean(G_loss_list), mean(D_loss_list), mean(reconstruction_loss_list)

    def train_one_scale(self, generator: Generator, discriminator: Discriminator, G_optimizer, D_optimizer,
                        real_image: torch.Tensor, noise_size: Tuple[int, int, int, int], pre_output,
                        reconstruction_loss_rate=100) -> \
            Tuple[torch.Tensor, float, float, float]:
        noise = self.make_noise(noise_size)
        fake_label = try_cuda(torch.zeros(noise_size[0])).long()
        real_label = try_cuda(torch.ones(noise_size[0])).long()

        # Adversarial loss of generator
        fake_image = generator(noise, pre_output)
        D_fake_out = discriminator(fake_image)
        G_adversarial_loss = F.cross_entropy(D_fake_out, real_label)
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(fake_image, real_image)
        G_total_loss = G_adversarial_loss + reconstruction_loss * reconstruction_loss_rate
        discriminator.zero_grad()
        G_optimizer.zero_grad(), D_optimizer.zero_grad()
        G_total_loss.backward()
        G_optimizer.step()

        # Adversarial loss of discriminator
        D_real_out = discriminator(real_image)
        D_loss_real, D_loss_fake = F.cross_entropy(D_real_out, real_label), F.cross_entropy(D_fake_out.detach(),
                                                                                            fake_label)
        D_loss = D_loss_real + D_loss_fake
        discriminator.zero_grad(), generator.zero_grad()
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        return fake_image.detach(), G_adversarial_loss.item(), D_loss.item(), reconstruction_loss.item()

    def make_noise(self, size: Tuple[int, int, int, int]):
        return try_cuda(torch.randn(size))

    def _to_eval_models(self, models: List[nn.Module]):
        for model in models:
            model.eval()

    def _to_train_models(self, models: List[nn.Module]):
        for model in models:
            model.train()

    def super_resolution(self, img: Image.Image, step: int = 2) -> Image.Image:
        img_tensor = ToTensor()(img)
        img_tensor = try_cuda(img_tensor.view((1,) + img_tensor.shape))
        self.G_list[-1].eval()
        super_img = img_tensor
        for i in range(step):
            super_img = self.G_list[-1].forward_image(super_img)
        return ToPILImage()(super_img[0].cpu().detach())

    def random_realistic_image(self, size_list: List[int]) -> Image.Image:
        for G in self.G_list:
            G.eval()
        pre_output = None
        for i, size in enumerate(size_list):
            noise = try_cuda(self.make_noise((1, 3, size, size)))
            pre_output = F.interpolate(pre_output, (size, size),
                                       mode="bilinear") if pre_output is not None else None
            pre_output = self.G_list[i](noise, pre_output)
        return ToPILImage()(pre_output[0].cpu().detach())


class SuperResolutionCallback:
    def __init__(self, dataset: Dataset, singan: SinGAN, base_img_path, out_dir: str, per_epoch: int):
        self._out_dir = out_dir
        self._base_img_path = base_img_path
        self._dataset = dataset
        self._singan = singan
        self._per_epoch = per_epoch

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        # data_len = len(self._dataset)
        # random_image_index = np.random.randint(0, data_len)
        # img = ToPILImage()(self._dataset[random_image_index])
        img = Image.open(self._base_img_path)
        super_resolution_image = self._singan.super_resolution(img)
        super_resolution_image.save(f"{self._out_dir}/epoch{epoch + 1}_sr.png")
        img.save(f"{self._out_dir}/epoch{epoch + 1}_base.png")


class RandomRealisticImageCallback:
    def __init__(self, dataset: Dataset, singan: SinGAN, base_img_path, out_dir: str, per_epoch: int,
                 size_list: List[int]):
        self._out_dir = out_dir
        self._base_img_path = base_img_path
        self._dataset = dataset
        self._singan = singan
        self._per_epoch = per_epoch
        self._size_list = size_list

    def __call__(self, epoch: int):
        if (epoch + 1) % self._per_epoch != 0:
            return
        img = Image.open(self._base_img_path)
        random_realistic_image = self._singan.random_realistic_image(self._size_list)
        img.save(f"{self._out_dir}/epoch{epoch + 1}_base.png")
        random_realistic_image.save(f"{self._out_dir}/epoch{epoch + 1}_rand.png")
