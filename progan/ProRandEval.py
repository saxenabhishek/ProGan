import torch
import matplotlib.pyplot as plt

import sys

sys.path.append("./progan")

import Definitions.proGen as model
import torchvision.utils as vutils


def walk(Genpath: str, numrows: int = 3, step: int = 4):
    print("Test : Walking in latent sapce")
    netG = model.ProGen(4, tanh=True)
    print("     * Model Made   ")
    loads = torch.load(Genpath)
    netG.load_state_dict(loads[":gen"])
    d = loads["currentLayerDepth"]
    a = loads["alpha"]
    print("     * Weights Loaded   ")

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


def walk2d(gen_path: str, save_dir, numrows: int, numcols: int, step: int, points: int) -> None:
    print("Test : Walking in latent sapce")
    netG = model.ProGen(4, tanh=True)
    print("     * Model Made   ")

    loads = torch.load(gen_path)
    netG.load_state_dict(loads[":gen"])

    d = loads["currentLayerDepth"]
    a = loads["alpha"]
    print("     * Weights Loaded   ")

    vec_shape = 512
    z = torch.randn(numcols * numrows, vec_shape)
    z2 = torch.randn(numcols * numrows, vec_shape)

    diff = z2 - z
    print("     * Initial Noise Created   ")
    for j in range(points):
        diff = z2 - z
        for i in range(step):
            img_vec = z + diff * (i / step)
            with torch.no_grad():
                img = netG(img_vec, depth=d, alpha=a)
                plt.imshow(vutils.make_grid(img, nrow=numrows, normalize=True, padding=2).permute(1, 2, 0))
                plt.axis(False)
                plt.savefig(save_dir + f"{j}-{i}.png", bbox_inches="tight", dpi=120)
        z = z2.clone()
        z2 = torch.randn(numcols * numrows, vec_shape)

    print("     * ALL Images genarated    ")


if __name__ == "__main__":
    walk2d("Parm_weig_CIFAR10_3depth.tar", "Cifar-10/", 6, 6, step=5, points=8)

