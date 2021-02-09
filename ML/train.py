"""

Training

"""

import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import Data
from .models import Generator, Discriminator, ResNetEncoder
from torchvision.utils import make_grid

# import matplotlib.pyplot as plt
from time import time

def show_tensor_images(image_tensor, num_images=64, size=(3, 64, 64)):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



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


def train_automate(epoch, path, split, vec_shape=1000, batch_size=64):
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

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    netG = netG.apply(weights_init)
    netD = netD.apply(weights_init)

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

       

        imagebatch, _  = next(iter(d_loaded))   

        with torch.no_grad():
            vector = netENC(imagebatch)
            fakeImage = netG(vector)

             
        show_tensor_images(fakeImage)
        show_tensor_images(imagebatch)
    
    return disF, disR, lossG, lossD
