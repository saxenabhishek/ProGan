"""
Custom layers
"""

import torch.nn as nn
from numpy import prod
import math


class pixelNorm(nn.Module):
    """
    It does this in short
    y = x/sqrt(mean(x**2) + ep)
    """

    def __init__(self):
        super(pixelNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


class ConstrainedLayer(nn.Module):
    def __init__(self, module, equalized=True, lrMul=1.0, initBiasToZero=True):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = self.getLayerNormalizationFactor(self.module) * lrMul

    def getLayerNormalizationFactor(self, x):
        size = x.weight.size()
        fan_in = prod(size[1:])

        return math.sqrt(2.0 / fan_in)

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqConv2d(ConstrainedLayer):
    def __init__(self, nChannelsPrevious, nChannels, kernelSize, stride=1, padding=0, bias=True, **kwargs):
        ConstrainedLayer.__init__(
            self, nn.Conv2d(nChannelsPrevious, nChannels, kernelSize, stride, padding=padding, bias=bias,), **kwargs
        )


class EqLinear(ConstrainedLayer):
    def __init__(self, nChannelsPrevious, nChannels, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Linear(nChannelsPrevious, nChannels, bias=bias), **kwargs)
