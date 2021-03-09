import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys

sys.path.append("./ML")

import Definitions.models as models
from Definitions.dataset import Data


def main(imgpath="Data", noise_dim=100, vec_shape=100, root="./ModelWeights/", numrows=5):

    netG = models.Generator(device="cpu", noise_dim=noise_dim, vec_shape=vec_shape)
    netENC = models.ResNetEncoder(vec_shape)

    netG.load_state_dict(torch.load(root + "Gen.pt"))
    netENC.load_state_dict(torch.load(root + "RES.pt"))

    netG.eval()
    netENC.eval()

    d = Data(path=imgpath, batch_size=numrows, size1=(225, 225), size2=(8, 8))

    d_loaded = DataLoader(d.folderdata, numrows, shuffle=True)

    # get one random batch of images
    BATCH = next(iter(d_loaded))
    imgs = BATCH["S1"]
    imgs2 = BATCH["S2"]
    with torch.no_grad():
        vector = netENC(imgs)
        fakeImages = netG(vector)

    _, ax = plt.subplots(2, numrows, squeeze=False, sharex=True, sharey=True, figsize=(8, 4))

    for i in range(numrows):
        ax[0, i].imshow((fakeImages[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2])
        ax[0, i].axis(False)

        ax[1, i].imshow((imgs2[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2])
        ax[1, i].axis(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()

