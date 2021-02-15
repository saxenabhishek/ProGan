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

from Definitions.ProGen import ProGen
from Definitions.ProDis import ProDis

from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from time import time


class train:
    def __init__(self, path, batch_size, split, savedir="ModelWeights", merge_samples_Const=1, loadmodel=False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.gen = ProGen().to(self.device)
        self.disc = ProDis().to(self.device)
        self.continuetraining = loadmodel
        # self.resnet = ResNetEncoder(vec_shape=vec_shape).to(self.device)
        # self.epochs = epochs
        # self.display_step = display_step
        self.root = savedir + "/"
        self.criterion = nn.BCEWithLogitsLoss()
        beta1 = 0.0
        lr = 0.003
        self.batch_size = batch_size

        self.currentSize = (4, 4)
        self.previousSize = (2, 2)
        self.currentLayerDepth = 0

        self.discopt = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, 0.999))
        self.genopt = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))
        data = Data(path=path, batch_size=batch_size, size1=self.currentSize, size2=self.previousSize, num_workers=1)
        self.trainloader, self.testloader, _ = data.getdata(split=split)

        self.alpha = 1
        self.alpha_unit_update = 1 / len(self.trainloader)  # / merge_samples_Const
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

        cur_step = 0
        mean_discriminator_loss = 0
        mean_generator_loss = 0
        test_noise = torch.rand(self.batch_size, 512).to(self.device)

        for epoch in range(epochs):
            print("training")
            for batch in tqdm(self.trainloader):
                ## training disc

                imageS1 = batch["S1"].to(self.device)
                imageS2 = F.interpolate(batch["S2"], scale_factor=2).to(self.device)

                if self.alpha < 1:
                    self.alpha += self.alpha_unit_update
                else:
                    self.alpha = 1

                real_image = (1 - self.alpha) * imageS2 + self.alpha * imageS1

                noise = torch.rand(self.batch_size, 512).to(self.device)

                self.discopt.zero_grad()

                discrealout = self.disc(real_image, self.currentLayerDepth, self.alpha)

                discfakeout = self.disc(
                    self.gen(noise, self.currentLayerDepth, self.alpha).detach(), self.currentLayerDepth, self.alpha
                )

                realdiscloss = self.criterion(discrealout, torch.ones_like(discrealout))
                fakediscloss = self.criterion(discfakeout, torch.zeros_like(discfakeout))

                totaldiscloss = (realdiscloss + fakediscloss) / 2
                totaldiscloss.backward()

                mean_discriminator_loss += totaldiscloss.item()
                self.discopt.step()

                ##trianing generator

                self.genopt.zero_grad()

                genout = self.disc(
                    self.gen(noise, self.currentLayerDepth, self.alpha), self.currentLayerDepth, self.alpha
                )

                genoutloss = self.criterion(genout, torch.ones_like(genout))

                genoutloss.backward()
                mean_generator_loss += genoutloss.item()
                self.genopt.step()

                # print(cur_step)
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, \t discriminator loss: {mean_discriminator_loss}"
                    )
                    fake = self.gen(test_noise, self.currentLayerDepth, self.alpha)
                    self.show_tensor_images(torch.cat((fake, real_image), 0))

                    self.discLosses.append(mean_discriminator_loss / display_step)
                    self.genLosses.append(mean_generator_loss / display_step)

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

            print("Saving weights")

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
    gan = train("./Data", 5, [1, 200, 0], "./ModelWeights",)
    gan.trainer(20, 1000)
    gan.plot_trainer()

