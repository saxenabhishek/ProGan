"""

Training

"""

import torch
import torch.nn as nn
import torch.optim as optim

import sys

sys.path.append("./ML")

from Definitions.dataset import Data
from Definitions.models import Generator, Discriminator, ResNetEncoder
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from time import time


class train:
    def __init__(
        self, path, epochs, batch_size, split, display_step=50, vec_shape=100, noisedim=100, savedir="ModelWeights"
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = Generator(device=self.device, noise_dim=noisedim, vec_shape=vec_shape).to(self.device)
        self.disc = Discriminator().to(self.device)
        self.resnet = ResNetEncoder(vec_shape=vec_shape).to(self.device)
        self.epochs = epochs
        self.display_step = display_step
        self.root = savedir + "/"
        self.criterion = nn.MSELoss()
        beta1 = 0.5
        lr = 0.002
        self.discopt = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, 0.999))
        self.genopt = optim.Adam(
            list(self.resnet.parameters()) + list(self.gen.parameters()), lr=lr, betas=(beta1, 0.999)
        )
        data = Data(path=path, batch_size=batch_size, size=(64, 64))
        self.trainloader, self.testloader, _ = data.getdata(split=split)

        self.gen = self.gen.apply(self.weights_init)
        self.disc = self.disc.apply(self.weights_init)

        self.discLosses = []
        self.genLosses = []

    def trainer(self):

        self.gen.train()
        self.disc.train()

        cur_step = 0
        display_step = self.display_step
        mean_discriminator_loss = 0
        mean_generator_loss = 0
        testimage = next(iter(self.testloader))
        testimage = testimage[0].to(self.device)

        for epoch in range(self.epochs):
            print("training")
            for image, _ in tqdm(self.trainloader):
                ## training disc

                image = image.to(self.device)
                self.discopt.zero_grad()
                discrealout = self.disc(image)
                vector = self.resnet(image)
                discfakeout = self.disc(self.gen(vector).detach())

                realdiscloss = self.criterion(discrealout, torch.ones_like(discrealout))
                fakediscloss = self.criterion(discfakeout, torch.zeros_like(discfakeout))

                totaldiscloss = (realdiscloss + fakediscloss) / 2
                totaldiscloss.backward()

                mean_discriminator_loss += totaldiscloss.item() / display_step
                self.discopt.step()

                ##trianing generator

                self.genopt.zero_grad()
                genout = self.disc(self.gen(vector))
                genoutloss = self.criterion(genout, torch.ones_like(genout))

                genoutloss.backward()
                mean_generator_loss += genoutloss.item() / display_step
                self.genopt.step()

                # print(cur_step)
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, \t discriminator loss: {mean_discriminator_loss}"
                    )
                    fake = self.gen(self.resnet(testimage))
                    self.show_tensor_images(fake)
                    self.show_tensor_images(testimage)

                    self.discLosses.append(mean_discriminator_loss)
                    self.genLosses.append(mean_generator_loss)

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                cur_step += 1

            print("Saving weights")

            torch.save(self.resnet.state_dict(), self.root + "RES.pt")
            torch.save(self.gen.state_dict(), self.root + "Gen.pt")
            torch.save(self.disc.state_dict(), self.root + "Dis.pt")

    def show_tensor_images(self, image_tensor, num_images=64, size=(3, 64, 64)):

        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis(False)
        plt.show()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def plot_trainer(self):
        assert len(self.discLosses) != 0 and len(self.genLosses) != 0
        plt.plot(self.discLosses, label="Discriminator Loss")
        plt.plot(self.genLosses, label="Generator Loss")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    train = train("../../fashiondata/img", epochs=1, batch_size=100, vec_shape=100)
    train.trainer()

