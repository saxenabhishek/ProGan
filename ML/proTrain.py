"""

Training

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys

sys.path.append("./ML")

from Definitions.dataset import Data
import math
from Definitions.proGen import ProGen
from Definitions.proDis import ProDis
from Definitions.loss import LSGAN, WGANGP

from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from time import time


class train:
    def __init__(
        self,
        path,
        batch_size,
        split,
        savedir="ModelWeights",
        merge_samples_Const=1,
        loadmodel=False,
        lr=[0.0001, 0.0001],
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.gen = ProGen(tanh=False)
        self.disc = ProDis()
        self.continuetraining = loadmodel

        self.root = savedir + "/"

        self.loss = WGANGP()
        beta1 = 0.0
        self.batch_size = batch_size

        self.currentSize = (4, 4)
        self.previousSize = (2, 2)
        self.currentLayerDepth = 0

        self.discopt = optim.Adam(self.disc.parameters(), lr=lr[1], betas=(beta1, 0.999))
        self.genopt = optim.Adam(self.gen.parameters(), lr=lr[0], betas=(beta1, 0.999))
        data = Data(path=path, batch_size=batch_size, size1=self.currentSize, size2=self.previousSize, num_workers=1)
        self.trainloader, self.testloader, _ = data.getdata(split=split)

        self.alpha = 1
        self.alpha_unit_update = 1 / len(self.trainloader) / merge_samples_Const
        print(self.alpha_unit_update)

        if self.continuetraining == False:
            self.disc = self.disc.apply(self.weights_init)
            self.gen = self.gen.apply(self.weights_init)
        else:
            print("laoding weights")
            self.gen.load_state_dict(torch.load(self.root + "Gen.pt"))  ## path  generator
            self.disc.load_state_dict(torch.load(self.root + "Dis.pt"))  ## path

        self.discLosses = []
        self.genLosses = []

    def trainer(self, epochs, display_step):

        self.gen.train()
        self.disc.train()

        self.gen = self.gen.to(self.device)
        self.disc = self.disc.to(self.device)

        cur_step = 0
        mean_discriminator_loss = 0
        mean_generator_loss = 0
        test_noise = torch.randn(self.batch_size, 512).to(self.device)

        for epoch in range(epochs):
            print("training")
            for batch in tqdm(self.trainloader):
                ## training disc

                torch.cuda.empty_cache()

                imageS1 = batch["S1"].to(self.device)
                imageS2 = F.interpolate(batch["S2"], scale_factor=2).to(self.device)

                if self.alpha < 1:
                    self.alpha += self.alpha_unit_update
                else:
                    self.alpha = 1

                real_image = (1 - self.alpha) * imageS2 + self.alpha * imageS1

                batch_size = real_image.shape[0]
                noise = torch.randn(batch_size, 512).to(self.device)

                self.discopt.zero_grad()
                self.genopt.zero_grad()

                discrealout = self.disc(real_image, self.currentLayerDepth, self.alpha)

                fake_image = self.gen(noise, self.currentLayerDepth, self.alpha).detach()
                discfakeout = self.disc(fake_image, self.currentLayerDepth, self.alpha)

                self.discLosses.append(
                    self.loss.disLoss(
                        discrealout,
                        discfakeout,
                        real=real_image,
                        fake=fake_image,
                        disc=self.disc,
                        depth=self.currentLayerDepth,
                        a=self.alpha,
                    )
                )
                mean_discriminator_loss += self.discLosses[-1]
                self.discopt.step()

                ##trianing generator

                self.genopt.zero_grad()
                self.discopt.zero_grad()

                noise = torch.randn(batch_size, 512).to(self.device)

                genout = self.disc(
                    self.gen(noise, self.currentLayerDepth, self.alpha), self.currentLayerDepth, self.alpha
                )
                self.genLosses.append(self.loss.genloss(genout))
                mean_generator_loss += self.genLosses[-1]
                self.genopt.step()

                # Evaluation
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"[{epoch}] Step {cur_step}: Generator loss: {mean_generator_loss /display_step}, \t discriminator loss: {mean_discriminator_loss/display_step}  a:{self.alpha}"
                    )
                    fake = self.gen(test_noise, self.currentLayerDepth, self.alpha)
                    self.show_tensor_images(torch.cat((fake, real_image), 0))

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

            print("Saving weights")

            torch.save(self.gen.state_dict(), self.root + "Gen.pt")
            torch.save(self.disc.state_dict(), self.root + "Dis.pt")

    def step_up(self):
        self.currentLayerDepth += 1

        self.previousSize = self.currentSize
        self.currentSize = (self.currentSize[0] * 2,) * 2

        print(self.previousSize, self.currentSize)

        self.trainloader.dataset.dataset.s1 = self.currentSize
        self.trainloader.dataset.dataset.s2 = self.previousSize
        self.alpha = 0

    def step_dn(self):
        self.currentLayerDepth -= 1

        self.currentSize = (self.currentSize[0] // 2,) * 2
        self.previousSize = (self.currentSize[0] // 2,) * 2

        print(self.previousSize, self.currentSize)

        self.trainloader.dataset.dataset.s1 = self.currentSize
        self.trainloader.dataset.dataset.s2 = self.previousSize
        self.alpha = 0

    def show_tensor_images(self, image_tensor):

        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        plt.figure(figsize=(5, 5))
        numImgs = image_tensor.shape[0]
        edgeNum = int(numImgs / int(math.sqrt(numImgs)))
        image_grid = make_grid(image_unflat, nrow=edgeNum)
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
    gan = train("./Data", 6, [1, 200, 0], "./ModelWeights", lr=[0.0003, 0.0001], merge_samples_Const=10)
    # gan.step_up()
    gan.trainer(5, 50)
    gan.step_up()
    gan.trainer(5, 50)
    # gan.step_up()
    # gan.trainer(3, 50)
    gan.plot_trainer()

