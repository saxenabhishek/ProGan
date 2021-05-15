import torch.nn as nn
import torch
import torch.nn.functional as F

import sys

sys.path.append("./progan/Definitions/")

from layers import pixelNorm, EqConv2d, EqLinear


class ProDis(nn.Module):
    def __init__(self, layer_dept: int = 5, Uper_feat_Exp: int = 9):
        super(ProDis, self).__init__()
        self.layer_depth = layer_dept
        self.Uper_feat_Exp = Uper_feat_Exp
        self.max_filter = 2 ** self.Uper_feat_Exp

        self.first_layer = nn.ModuleList(
            [
                self.fromrgb(2 ** (self.Uper_feat_Exp - i + 4)) if i > 3 else self.fromrgb(2 ** (self.Uper_feat_Exp))
                for i in range(self.layer_depth + 1)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                self.block(2 ** (self.Uper_feat_Exp - i - 1 + 4), 2 ** (self.Uper_feat_Exp - i + 4))
                if i > 3
                else self.block(2 ** (self.Uper_feat_Exp), 2 ** (self.Uper_feat_Exp))
                for i in range(self.layer_depth)
            ]
        )

        self.last_layer = nn.Sequential(
            EqConv2d(self.max_filter + 1, self.max_filter, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
            EqConv2d(self.max_filter, self.max_filter, 4, 1),
            nn.LeakyReLU(0.2),
        )

        self.fc = EqLinear(self.max_filter, 1)

    def fromrgb(self, maps):
        return EqConv2d(3, maps, 1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            EqConv2d(in_ch, out_ch, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
            EqConv2d(out_ch, out_ch, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

    def miniBatchSTD(self, x):
        # std deviation over differnt batches
        std = torch.std(x, 0)
        # average of that
        avg = torch.mean(std)

        expanded = avg.expand_as(x[:, :1])
        return torch.cat([x, expanded], 1)

    def forward(self, x, depth, alpha=0.5):
        assert depth <= self.layer_depth

        after_x = x

        x = self.first_layer[depth](x)
        x = F.leaky_relu(x, 0.2)

        if depth != 0:
            x = self.blocks[depth - 1](x)
            if alpha != 1:
                after_x = F.avg_pool2d(after_x, 2)
                after_x = self.first_layer[depth - 1](after_x)
                after_x = F.leaky_relu(after_x, 0.2)
                x = (1 - alpha) * after_x + alpha * x

        for j in range(depth - 2, -1, -1):
            x = self.blocks[j](x)

        x = self.miniBatchSTD(x)
        x = self.last_layer(x).squeeze()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    d = ProDis()
    print(d)
    out = d(torch.rand(12, 3, 128, 128), 5)
    print(out.shape)
