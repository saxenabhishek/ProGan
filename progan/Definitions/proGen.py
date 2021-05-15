import torch.nn as nn
import torch
import torch.nn.functional as F

import sys

sys.path.append("./progan/Definitions/")

from layers import pixelNorm, EqConv2d, EqLinear


class ProGen(nn.Module):
    def __init__(self, layer_dept: int = 5, Uper_feat_Exp: int = 9, tanh: bool = False):
        super(ProGen, self).__init__()
        self.layer_depth = layer_dept
        self.Uper_feat_Exp = Uper_feat_Exp
        self.max_filter = 2 ** self.Uper_feat_Exp
        self.tanh = tanh

        self.fc = EqLinear(self.max_filter, self.max_filter)
        self.first_layer = nn.Sequential(
            nn.Conv2d(self.max_filter, self.max_filter, 4, 1, 3,),
            nn.LeakyReLU(0.2),
            EqConv2d(self.max_filter, self.max_filter, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
        )

        self.blocks = nn.ModuleList(
            [
                self.block(2 ** (self.Uper_feat_Exp - i + 4), 2 ** (self.Uper_feat_Exp - i - 1 + 4))
                if i > 3
                else self.block(2 ** (self.Uper_feat_Exp), 2 ** (self.Uper_feat_Exp))
                for i in range(self.layer_depth)
            ]
        )

        self.last_layer = nn.ModuleList(
            [
                self.torgb(2 ** (self.Uper_feat_Exp - i + 4)) if i > 4 else self.torgb(2 ** (self.Uper_feat_Exp))
                for i in range(self.layer_depth + 1)
            ]
        )

    def torgb(self, maps):
        return EqConv2d(maps, 3, 1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            EqConv2d(in_ch, out_ch, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
            EqConv2d(out_ch, out_ch, 3, 1, 1),
            pixelNorm(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, depth, alpha=1):
        assert depth <= self.layer_depth
        bs = x.shape[0]
        before_x = None

        x = self.fc(x).reshape(bs, self.max_filter, 1, 1)
        x = self.first_layer(x)

        for i in range(depth):
            before_x = x
            x = self.blocks[i](before_x)

        x = self.last_layer[depth](x)

        if depth != 0 and alpha < 1:
            before_x = self.last_layer[depth - 1](F.interpolate(before_x, scale_factor=2))
            x = (1 - alpha) * before_x + alpha * x

        if self.tanh:
            return torch.tanh(x)
        else:
            return x


if __name__ == "__main__":
    g = ProGen(layer_dept=6, Uper_feat_Exp=9)
    print(g)
    out = g(torch.rand(1, 512), 6, 0.5)
    print(out.shape)
