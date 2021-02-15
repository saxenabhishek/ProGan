import torch.nn as nn
import torch
import torch.nn.functional as F


class ProDis(nn.Module):
    def __init__(self, layer_dept=5):
        super(ProDis, self).__init__()
        self.layer_depth = layer_dept
        self.Uper_feat_Exp = 8
        self.max_filter = 2 ** self.Uper_feat_Exp

        self.first_layer = nn.ModuleList(
            [self.fromrgb(2 ** (self.Uper_feat_Exp - i)) for i in range(self.layer_depth + 1)]
        )

        self.gen_blocks = nn.ModuleList(
            [
                self.block(2 ** (self.Uper_feat_Exp - i - 1), 2 ** (self.Uper_feat_Exp - i))
                for i in range(self.layer_depth)
            ]
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.max_filter + 1, self.max_filter, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.max_filter, self.max_filter, 4, 1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(self.max_filter, 1)

    def fromrgb(self, maps):
        return nn.Conv2d(3, maps, 1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
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
            x = self.gen_blocks[depth - 1](x)
            if alpha != 1:
                after_x = F.avg_pool2d(after_x, 2)
                after_x = self.first_layer[depth - 1](after_x)
                after_x = F.leaky_relu(after_x, 0.2)
                x = (1 - alpha) * after_x + alpha * x

        for j in range(depth - 2, -1, -1):
            x = self.gen_blocks[j](x)

        x = self.miniBatchSTD(x)
        x = self.last_layer(x).squeeze()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    d = ProDis()

    print(d)
    out = d(torch.rand(12, 3, 128, 128), 5)

    print(out.shape)
