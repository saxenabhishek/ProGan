import torch.nn as nn
import torch
import torch.nn.functional as F


class ProGen(nn.Module):
    def __init__(self, layer_dept=5, Uper_feat_Exp=9):
        super(ProGen, self).__init__()
        self.layer_depth = layer_dept
        self.Uper_feat_Exp = Uper_feat_Exp
        self.max_filter = 2 ** self.Uper_feat_Exp

        self.fc = nn.Linear(self.max_filter, self.max_filter)
        self.first_layer = nn.Sequential(
            nn.Conv2d(self.max_filter, self.max_filter, 4, 1, 3),
            nn.BatchNorm2d(self.max_filter),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.max_filter, self.max_filter, 3, 1, 1),
            nn.BatchNorm2d(self.max_filter),
            nn.LeakyReLU(0.2),
            # nn.LocalResponseNorm(self.max_filter, alpha=1, beta=0.5, k=1e-8),
        )

        self.gen_blocks = nn.ModuleList(
            [
                self.block(2 ** (self.Uper_feat_Exp - i), 2 ** (self.Uper_feat_Exp - i - 1))
                for i in range(self.layer_depth)
            ]
        )

        self.last_layer = nn.ModuleList(
            [self.torgb(2 ** (self.Uper_feat_Exp - i)) for i in range(self.layer_depth + 1)]
        )

    def torgb(self, maps):
        return nn.Conv2d(maps, 3, 1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
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
            x = self.gen_blocks[i](before_x)

        x = self.last_layer[depth](x)

        if depth != 0 and alpha < 1:
            before_x = self.last_layer[depth - 1](F.interpolate(before_x, scale_factor=2))
            return torch.tanh((1 - alpha) * before_x + alpha * x)
        else:
            return torch.tanh(x)


if __name__ == "__main__":
    g = ProGen()
    print
    out = g(torch.rand(1, 512), 4, 0.5)
    print(out.shape)
