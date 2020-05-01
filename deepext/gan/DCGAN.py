from __future__ import print_function
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
from statistics import mean


class DCGAN:
    def __init__(self, latent_variable_dim: int, lr: int = 0.0001):
        self.generator = Generator(latent_variable_dim).to("cuda:0")
        self.discriminator = Discriminator().to("cuda:0")
        self.latent_variable_dim = latent_variable_dim
        self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                              lr=lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                  lr=lr, betas=(0.5, 0.999))
        self.loss_func = nn.BCEWithLogitsLoss()

    def train_step(self, data_loader: DataLoader):
        batch_size = data_loader.batch_size
        real_labels, fake_labels = torch.ones(batch_size).to("cuda:0"), torch.zeros(batch_size).to("cuda:0")
        generator_loss_list, discriminator_loss_list = [], []
        for real_img, _ in tqdm.tqdm(data_loader, 0):
            batch_len = len(real_img)

            # Generate
            real_img = real_img.to("cuda:0")
            noise = torch.randn(batch_len, self.latent_variable_dim, 1, 1).to("cuda:0")
            fake_img = self.generator(noise)
            fake_img_temp = fake_img.detach()
            # 偽画像のDiscriminator loss
            out = self.discriminator(fake_img)
            generator_loss = self.loss_func(out, real_labels[: batch_len])
            generator_loss_list.append(generator_loss.item())
            self.discriminator.zero_grad()
            self.generator.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            # Discriminator
            real_out = self.discriminator(real_img)
            discriminator_loss_real = self.loss_func(real_out, real_labels[: batch_len])
            fake_out = self.discriminator(fake_img_temp)
            discriminator_loss_fake = self.loss_func(fake_out, fake_labels[: batch_len])
            # 実画像と偽画像のロスを合計
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss_list.append(discriminator_loss.item())
            self.discriminator.zero_grad()
            self.generator.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        return mean(generator_loss_list), mean(discriminator_loss_list)


class Generator(nn.Module):
    def __init__(self, latent_variable_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_variable_dim, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).squeeze()
