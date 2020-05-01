import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from deepext import *

if __name__ == "__main__":
    # === 1. データの読み込み ===
    # datasetrの準備
    dataset = datasets.ImageFolder("D:/dataset/gan/sample-data/",
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
    batch_size = 64

    # dataloaderの準備
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    var = 10
    check_z = torch.randn(64, var, 1, 1).to("cuda:0")
    dcgan = DCGAN(var)
    for epoch in range(2000):
        print("epoch : " + str(epoch))
        print(dcgan.train_step(data_loader))

        # 訓練途中のモデル・生成画像の保存
        if epoch % 10 == 0:
            torch.save(
                dcgan.generator.state_dict(),
                "./g/G_{:03d}.prm".format(epoch),
                pickle_protocol=4)
            torch.save(
                dcgan.discriminator.state_dict(),
                "./d/D_{:03d}.prm".format(epoch),
                pickle_protocol=4)
            # check_z = torch.Tensor([0, 0, 0, 0.8, 0.2, 0, 0, 0, 0, 0]).view((1, -1, 1, 1)).to("cuda:0")
            generated_img = dcgan.generator(check_z)
            save_image(generated_img,
                       "./i/{:03d}.jpg".format(epoch))
