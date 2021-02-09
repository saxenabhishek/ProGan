"""

Training

"""

import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import Data
from .models import Generator, Discriminator, ResNetEncoder


def pass_element(img, netD, netG, netENC, optD, optG, criterion):

    netD.zero_grad()

    # Real Image for Disc
    D1 = netD(img)

    label = torch.ones_like(D1)
    err_real = criterion(D1, label)
    err_real.backward()

    # Fake Image for Disc
    vector = netENC(img)
    fakeImage = netG(vector)
    D2 = netD(fakeImage.detach())

    label = torch.zeros_like(D2)
    err_fake = criterion(D2, label)
    err_fake.backward()

    optD.step()

    # zero grn and resNet
    netENC.zero_grad()
    netG.zero_grad()

    # Genarated Image for Gen
    label = torch.ones_like(fakeImage)
    err_GenReal = criterion(fakeImage, label)
    err_GenReal.backward()

    optG.step()

    return D1.mean().item(), D2.mean().item()


def train_step(dataloader, device, netD, netG, netENC, optD, optG, criterion):

    disF = []
    disR = []
    print("Starting Training Loop...")
    for i, data in enumerate(dataloader):
        data_image = data[0]
        _ = data[1]

        img_Device = data_image.to(device)
        loss = pass_element(
            img_Device, netD, netG, netENC, optD, optG, criterion
        )

        print(i)

        disF.append(loss[1])
        disR.append(loss[0])

    return


def train_automate(epoch):
    vec_shape = 1000
    batch_size = 12

    d = Data("Data_small", batch_size=batch_size, size=(64, 64))
    split = [2, 1, 0]
    d_loaded, _, _ = d.getdata(split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    netG = Generator(
        device=device, noisedim=500, batch_size=batch_size, vec_shape=vec_shape
    )
    netD = Discriminator()
    netRes = ResNetEncoder(vec_shape)

    netG = netG.to(device)
    netD = netD.to(device)
    netRes = netRes.to(device)

    lr = 0.002
    beta1 = 0.5
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = optim.Adam(
        list(netRes.parameters()) + list(netG.parameters()),
        lr=lr,
        betas=(beta1, 0.999),
    )

    criterion = nn.BCEWithLogitsLoss()

    for i in range(epoch):
        train_step(d_loaded, device, netD, netG, netRes, optD, optG, criterion)
