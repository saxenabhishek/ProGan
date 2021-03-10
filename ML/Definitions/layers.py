"""
Custom layers
"""

import torch.nn as nn


class pixelNorm(nn.Module):
    """
    It does this in short
    y = x/sqrt(mean(x**2) + ep)
    """

    def __init__(self):
        super(pixelNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

