import torch
import matplotlib.pyplot as plt

import sys

sys.path.append("./progan")

import Definitions.proGen as model


def main(Genpath="C:\\Users\\as712\\Downloads\\Gen (5).pt", numrows=3, step=4):
    print("Test : Walking in latent sapce")
    netG = model.ProGen(4, tanh=True)
    print("     * Model Made   ")
    loads = torch.load(Genpath)
    netG.load_state_dict(loads[":gen"])
    d = loads["currentLayerDepth"]
    a = loads["alpha"]
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
    img = netG(z_all, depth=d, alpha=a).detach()
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
    main("Parm_weig_CIFAR10_3depth.tar", 5, 5)

