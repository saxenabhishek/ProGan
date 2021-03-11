import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys

sys.path.append("./ML")

import Definitions.proGen as model


def main(imgpath="Data", Genpath="ModelWeights/Gen.pt", numrows=3, step=4, d=1):
    print("Test : Walking in latent sapce")
    netG = model.ProGen(tanh=True)
    netG.load_state_dict(torch.load(Genpath))
    print("     * Weights Loaded   ")
    # netG.eval()

    vec_shape = 512
    z = torch.randn(numrows, vec_shape)
    z2 = torch.randn(numrows, vec_shape)

    diff = z2 - z

    z_all = z.clone()
    for i in range(step):
        inertia = z + diff * (i / step)
        z_all = torch.cat((inertia, z_all))

    print("     * Noise Created   ")
    img = netG(z_all, depth=d).detach()
    _, ax = plt.subplots(numrows, step + 1, sharex=True, sharey=True)
    print("     * Images genarated    ")
    for i in range(numrows):
        for ii in range(step + 1):
            ax[i, ii].imshow((img[(numrows * ii) + i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2])
            ax[i, ii].axis(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    print("     * ploted Images      ")
    plt.show()


if __name__ == "__main__":
    main()

