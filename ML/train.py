"""

Training

"""

import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import Data
from .models import Generator, Discriminator, ResNetEncoder

# import matplotlib.pyplot as plt
from time import time


def pass_element(img, netD, netG, netENC, optD, optG, criterion):
    # || Disc
    netD.zero_grad()

    # Real Image for Disc
    D1 = netD(img)

    label = torch.ones_like(D1)
    err_real = criterion(D1, label)

    # Fake Image for Disc
    vector = netENC(img)
    fakeImage = netG(vector)

    D2 = netD(fakeImage.detach())

    label = torch.zeros_like(D2)
    err_fake = criterion(D2, label)

    err_totalD = (err_real + err_fake) / 2
    err_totalD.backward()

    optD.step()

    # || Gen
    # zero grn and resNet
    netENC.zero_grad()
    netG.zero_grad()

    # Genarated Image for Gen
    D3 = netD(fakeImage)
    label = torch.ones_like(D3)
    err_GenReal = criterion(D3, label)
    err_GenReal.backward()

    optG.step()

    return (
        D1.mean().item(),
        D3.mean().item(),
        err_totalD.item(),
        err_GenReal.item(),
    )


def train_step(
    dataloader,
    device,
    netD,
    netG,
    netENC,
    optD,
    optG,
    criterion,
    disF,
    disR,
    lossG,
    lossD,
):

    for i, data in enumerate(dataloader):
        # st = time()
        data_image = data[0]

        # Binary class Male or female
        _ = data[1]

        img_Device = data_image.to(device)

        loss = pass_element(
            img_Device, netD, netG, netENC, optD, optG, criterion
        )

        disR.append(loss[0])
        disF.append(loss[1])
        lossD.append(loss[2])
        lossG.append(loss[3])

        if i % 50 == 0:
            print(
                f"|{i}| RealD {round(loss[0],4)}, FakeD {round(loss[1],4)}",
                end=",",
            )
            print(f"lossD {round(loss[2],4)}, lossG {round(loss[3],4)}")
            # st = 0


def train_automate(epoch, path):
    vec_shape = 1000
    batch_size = 8
    split = [1, 2, 0]

    d = Data(path, batch_size=batch_size, size=(64, 64))
    d_loaded, _, _ = d.getdata(split)

    print(len(d_loaded))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"  # overwrite to CPU for tests

    netG = Generator(
        device=device, noisedim=500, batch_size=batch_size, vec_shape=vec_shape
    )
    netD = Discriminator()
    netENC = ResNetEncoder(vec_shape)

    netG = netG.to(device)
    netD = netD.to(device)
    netENC = netENC.to(device)

    lr = 0.002
    beta1 = 0.5

    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = optim.Adam(
        list(netENC.parameters()) + list(netG.parameters()),
        lr=lr,
        betas=(beta1, 0.999),
    )

    criterion = nn.BCEWithLogitsLoss()
    disF = []
    disR = []
    lossG = []
    lossD = []

    print("Starting Training Loop...")
    starting_time = time()
    for i in range(epoch):
        print(f"[Epoch {i + 1}]")

        train_step(
            d_loaded,
            device,
            netD,
            netG,
            netENC,
            optD,
            optG,
            criterion,
            disF,
            disR,
            lossG,
            lossD,
        )
    print(f"total Time : {time() - starting_time}")
    root = "./ModelWeights/"
    torch.save(netENC.state_dict(), root + "RES.pt")
    torch.save(netG.state_dict(), root + "Gen.pt")
    torch.save(netD.state_dict(), root + "Dis.pt")

    # netG = netG.to("cpu")
    # netD = netD.to("cpu")
    # netENC = netENC.to("cpu")
    # netG.device = "cpu"

    # img = d.folderdata[20][0].unsqueeze(0)

    # with torch.no_grad():
    #     vector = netENC(img)
    #     fakeImage = netG(vector)

    # plt.imshow(fakeImage.permute(0, 2, 3, 1)[0])
    # plt.show()
    return disF, disR, lossG, lossD
