from torch import nn
from torchvision import models
import torch


class ResNetEncoder(nn.Module):
    def __init__(self, vec_shape):
        super(ResNetEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, vec_shape)

    def forward(self, image):
        return self.model(image)


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()

        self.device = kwargs["device"]
        # self.noisedim = kwargs["noise_dim"]
        # self.vector_shape = kwargs["vec_shape"]
        self.input_shape = kwargs["vec_shape"] + kwargs["noise_dim"]

        self.gen = nn.Sequential(
            self.genBlock(input_channels=self.input_shape, hidden_size=512, kernel_size=4, stride=1, padding=0,),
            self.genBlock(input_channels=512, hidden_size=350, kernel_size=4, stride=2, padding=1,),
            self.genBlock(input_channels=350, hidden_size=250, kernel_size=4, stride=2, padding=1,),
            self.genBlock(input_channels=250, hidden_size=150, kernel_size=4, stride=2, padding=1,),
            self.genBlock(
                input_channels=150,
                hidden_size=3,
                kernel_size=4,
                stride=2,
                padding=1,
                last_layer=True,  # final layer returning tanh
            ),
        )

    def genBlock(
        self, input_channels, hidden_size, kernel_size, stride, padding, last_layer=False,
    ):
        if not last_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                ),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                ),
                nn.Tanh(),
            )

    def genInput(self):
        return self.encodedVec.view(len(self.encodedVec), self.encodedVec.shape[1], 1, 1)

    def concat(self, batch_size):
        self.inputnoise = self.makeNoise(batch_size)
        encoded = torch.cat([self.feat, self.inputnoise], dim=1)
        return encoded

    def makeNoise(self, batch_size):
        return torch.randn(batch_size, self.noise_dim, device=self.device)

    def forward(self, feat):
        batch_size = feat.shape[0]
        self.feat = feat

        self.encodedVec = self.concat(batch_size)

        self.genIn = self.genInput()
        return self.gen(self.genIn)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.discBlock(inputChannels=3, outputChannels=128, first_layer=True),
            self.discBlock(inputChannels=128, outputChannels=256),
            self.discBlock(inputChannels=256, outputChannels=512),
        )

    def discBlock(self, inputChannels, outputChannels, first_layer=False):
        if first_layer:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=inputChannels,
                    out_channels=outputChannels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=inputChannels,
                    out_channels=outputChannels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(outputChannels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, inp):
        return self.disc(inp)


def main():
    vec_shape = 1000
    batch_size = 128
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    resNet = ResNetEncoder(vec_shape)
    resNet = resNet.to(device)
    gen = Generator(device=device, noise_dim=500, vec_shape=vec_shape)
    gen = gen.to(device)

    disc = Discriminator()

    for i in range(2):
        print(gen(resNet(torch.randn(batch_size, 3, 64, 64))).shape)
        print(disc(gen(resNet(torch.randn(batch_size, 3, 64, 64, device=device)))).shape)


if __name__ == "__main__":
    main()
