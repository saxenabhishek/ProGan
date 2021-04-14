"""
Training
"""

import math
import sys
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid

sys.path.append("./ML")

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from Definitions.dataset import Data
from Definitions.loss import WGANGP
from Definitions.proDis import ProDis
from Definitions.proGen import ProGen


class train:
    currentSize = (4, 4)
    previousSize = (2, 2)
    currentLayerDepth = 0
    epNUM = 0
    alpha = 1.0
    loss = WGANGP()

    def __init__(
        self,
        path: str,
        batch_size: int,
        split: list,
        savedir: str = "ModelWeights/",
        imagedir: str = "ModelWeights/img",
        merge_samples_Const: int = 1,
        lr: list = [0.0001, 0.0001],
        loadmodel: bool = False,
        PlotInNotebook: bool = False,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        print("Making models")
        self.gen = ProGen(tanh=False).to(self.device)
        self.disc = ProDis().to(self.device)
        self.test_noise = torch.randn(9, 512).to(self.device)

        print("Making optimizers")
        beta1 = 0.0
        self.discopt = optim.Adam(self.disc.parameters(), lr=lr[1], betas=(beta1, 0.999))
        self.genopt = optim.Adam(self.gen.parameters(), lr=lr[0], betas=(beta1, 0.999))

        print("Making Dataloader")
        data = Data(path=path, batch_size=batch_size, size1=self.currentSize, size2=self.previousSize, num_workers=1)
        self.trainloader, self.testloader, _ = data.getdata(split=split)

        self.alpha_speed = 1 / len(self.trainloader) / merge_samples_Const
        self.root = savedir
        self.rootimg = imagedir
        self.continuetraining: bool = loadmodel
        self.PlotInNotebook: bool = PlotInNotebook
        self.batch_size = batch_size

        self.losses: dict = {
            "disc": [],
            "gen": [],
            "probReal": [],
            "probFake": [],
        }

        if self.continuetraining:
            print("loading weights")
            self.loadValues()
            self.setImageSize()

    def trainer(self, epochs: int, display_step: int):

        self.gen.train()
        self.disc.train()

        cur_step = 0

        for epoch in range(epochs):
            print("training")
            for batch in tqdm(self.trainloader):
                ## training disc

                torch.cuda.empty_cache()

                # get images in 2 sizes
                imageS1 = batch["S1"].to(self.device)
                imageS2 = F.interpolate(batch["S2"], scale_factor=2).to(self.device)

                # update value of alpha if neccesary.
                if self.alpha < 1:
                    self.alpha += self.alpha_speed
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

                self.losses["probReal"].append(discrealout[:, 0].mean().item())
                self.losses["probFake"].append(discfakeout[:, 0].mean().item())

                self.losses["disc"].append(
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
                self.discopt.step()

                ##trianing generator

                self.genopt.zero_grad()
                self.discopt.zero_grad()

                noise = torch.randn(batch_size, 512).to(self.device)

                genout = self.disc(
                    self.gen(noise, self.currentLayerDepth, self.alpha), self.currentLayerDepth, self.alpha
                )
                self.losses["gen"].append(self.loss.genloss(genout))
                self.genopt.step()

                # Evaluation
                if cur_step % display_step == 0 and cur_step > 0:
                    print(" ")
                    print(f"\n ep{self.epNUM} | ")
                    for i in self.losses:
                        print(f" {i} : {self.movingAverage(i,display_step)}", end=" ")
                    print(f" Alpha : {self.alpha} ")

                    fake = self.gen(self.test_noise, self.currentLayerDepth, self.alpha)
                    self.show_tensor_images(torch.cat((fake, real_image[:9]), 0), cur_step)

                cur_step += 1

            print("Saving weights")
            self.save_weight()

            self.epNUM += 1

    def movingAverage(self, lossname: str, stepSize: int):
        vals = self.losses[lossname][-stepSize:]
        return sum(vals) / len(vals)

    def save_weight(self):
        torch.save(
            {
                ":gen": self.gen.state_dict(),
                ":disc": self.disc.state_dict(),
                ":discopt": self.discopt.state_dict(),
                ":genopt": self.genopt.state_dict(),
                "epNUM": self.epNUM,
                "alpha": self.alpha,
                "alpha_speed": self.alpha_speed,
                "currentSize": self.currentSize,
                "previousSize": self.previousSize,
                "currentLayerDepth": self.currentLayerDepth,
                "losses": self.losses,
            },
            self.root + "/Parm_weig.tar",
        )

    def loadValues(self):
        checkpoint = torch.load(self.root + "/Parm_weig.tar", map_location=self.device)
        for i in checkpoint:
            print("Loading ", i)
            if ":" in i:
                attr = i[1:]  # remove the ':'
                getattr(self, attr).load_state_dict(checkpoint[i])
            else:
                if i != "losses":
                    print(checkpoint[i])
                setattr(self, i, checkpoint[i])

    def setImageSize(self):
        self.trainloader.dataset.dataset.s1 = self.currentSize
        self.trainloader.dataset.dataset.s2 = self.previousSize

    def step_up(self):
        self.currentLayerDepth += 1

        self.previousSize = self.currentSize
        self.currentSize = (self.currentSize[0] * 2,) * 2

        print("Increasing size", self.previousSize, self.currentSize)
        self.setImageSize()
        self.alpha = 0

    def step_dn(self):
        self.currentLayerDepth -= 1

        self.currentSize = (self.currentSize[0] // 2,) * 2
        self.previousSize = (self.currentSize[0] // 2,) * 2

        print("Decreasing size", self.previousSize, self.currentSize)
        self.setImageSize()
        self.alpha = 0

    def show_tensor_images(self, image_tensor, step: int):
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        numImgs = image_tensor.shape[0]
        edgeNum = int(numImgs / int(math.sqrt(numImgs)))
        image_grid = make_grid(image_unflat, nrow=edgeNum)
        plt.title(str(self.epNUM) + "_" + str(self.currentLayerDepth) + "_" + str(self.alpha))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis(False)
        if self.PlotInNotebook:
            plt.show()
        else:
            plt.savefig(self.rootimg + "/" + str(self.epNUM) + "_" + str(step) + ".png", bbox_inches="tight")
            plt.clf()

    def plot_trainer(self):
        for i in self.losses:
            if "prob" not in i:
                plt.plot(self.losses[i], label=i)
        plt.legend()
        if self.PlotInNotebook:
            plt.show()
        else:
            plt.savefig(self.rootimg + "/" + "loss" + "_" + str(self.epNUM) + ".png")
            plt.clf()

        for i in self.losses:
            if "prob" in i:
                plt.plot(self.losses[i], label=i)
        plt.legend()
        if self.PlotInNotebook:
            plt.show()
        else:
            plt.savefig(self.rootimg + "/" + "Prob" + "_" + str(self.epNUM) + ".png")
            plt.clf()


if __name__ == "__main__":
    st = time()
    gan = train(
        "Data",
        128,
        [20, 80, 0],
        "D:\Projects\ProGan\ModelWeights",
        lr=[0.001, 0.001],
        merge_samples_Const=20,
        loadmodel=True,
        PlotInNotebook=False,
    )
    gan.trainer(20, 100)
    print(time() - st)

