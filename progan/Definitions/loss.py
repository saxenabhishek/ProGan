"""
loss
"""
import torch.nn as nn
import torch


class LSGAN:
    criterion = nn.MSELoss()

    def disLoss(self, discrealout, discfakeout, **kwarg):
        realdiscloss = self.criterion(discrealout, torch.ones_like(discrealout) * 0.9)
        fakediscloss = self.criterion(discfakeout, torch.ones_like(discfakeout) * 0.1)

        totaldiscloss = (realdiscloss + fakediscloss) / 2
        totaldiscloss.backward()

        return totaldiscloss.item()

    def genloss(self, genout):
        genoutloss = self.criterion(genout, torch.ones_like(genout) * 0.9)
        genoutloss.backward()
        return genoutloss.item()


class WGANGP:
    weight = 5
    epsilonD = 0.001

    def disLoss(self, dreal, dfake, **kwarg):
        D = kwarg["disc"]
        depth = kwarg["depth"]
        a = kwarg["a"]

        real = kwarg["real"]
        fake = kwarg["fake"]

        epsilonLoss = (dreal[:, 0] ** 2).sum() * self.epsilonD
        epsilonLoss.backward(retain_graph=True)

        crit = dfake[:, 0].sum() - dreal[:, 0].sum()
        crit.backward()

        batchSize = real.size(0)
        alpha = torch.rand(batchSize, 1)
        alpha = alpha.expand(batchSize, int(real.nelement() / batchSize)).contiguous().view(real.size())
        alpha = alpha.to(real.device)
        interpolates = alpha * real + ((1 - alpha) * fake)

        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        decisionInterpolate = D(interpolates, depth, a)
        decisionInterpolate = decisionInterpolate[:, 0].sum()

        gradients = torch.autograd.grad(
            outputs=decisionInterpolate, inputs=interpolates, create_graph=True, retain_graph=True
        )

        gradients = gradients[0].view(batchSize, -1)
        gradients = (gradients * gradients).sum(dim=1).sqrt()
        gradient_penalty = (((gradients - 1.0) ** 2)).sum() * self.weight

        gradient_penalty.backward()

        return gradient_penalty.detach().item() + crit.detach().item() + epsilonLoss.detach().item()

    def genloss(self, genout):
        crit = -genout[:, 0].sum()
        crit.backward()
        return crit.detach().item()


def main():
    WGANGP()


if __name__ == "__main__":
    main()
