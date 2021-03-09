import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys

sys.path.append("./ML")

import Definitions.models as models
from Definitions.dataset import Data


def main(imgpath="Data", noise_dim=1000, vec_shape=1000, root="./Landmarks_weights/", numrows=5):

    netG = models.Generator(device="cpu", noise_dim=noise_dim, vec_shape=vec_shape)
    netG.load_state_dict(torch.load(root + "Gen.pt"))
    # netG.eval()

    z = torch.rand(numrows, vec_shape)
    z2 = torch.rand(numrows, vec_shape)

    diff = z2 - z
    step = 10

    z_all = z.clone()
    for i in range(step):
        inertia = z + diff * (i / step)
        z_all = torch.cat((inertia, z_all))

    img = netG(z_all).detach()
    _, ax = plt.subplots(numrows, step + 1, sharex=True, sharey=True)

    for i in range(numrows):
        for ii in range(step + 1):
            ax[i, ii].imshow((img[(numrows * ii) + i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2])
            ax[i, ii].axis(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()

