import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ML.models
from ML.dataset import Data

if __name__ == "__main__":
    vec_shape = 1000
    batch_size = 8

    root = "./ModelWeights/"

    device = "cpu"
    netG = ML.models.Generator(
        device=device, noise_dim=500, vec_shape=vec_shape
    )

    netD = ML.models.Discriminator()
    netENC = ML.models.ResNetEncoder(vec_shape)

    netG.load_state_dict(torch.load(root + "Gen.pt"))
    netD.load_state_dict(torch.load(root + "Dis.pt"))
    netENC.load_state_dict(torch.load(root + "RES.pt"))

    # netG.eval()
    # netD.eval()
    # netENC.eval()

    d = Data("Data", batch_size=batch_size, size=(64, 64))

    bs = 4
    d_loaded = DataLoader(d.folderdata, bs, shuffle=True)

    imgs = next(iter(d_loaded))[0]

    with torch.no_grad():
        vector = netENC(imgs)
        fakeImages = netG(vector)

    _, ax = plt.subplots(
        2, bs, squeeze=False, sharex=True, sharey=True, figsize=(8, 4)
    )

    for i in range(bs):
        ax[0, i].imshow(
            (fakeImages[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2]
        )
        ax[0, i].axis(False)

        ax[1, i].imshow(
            (imgs[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2]
        )
        ax[1, i].axis(False)
    plt.show()
