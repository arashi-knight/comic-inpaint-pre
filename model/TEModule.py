from torch import nn


class TEModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TEModule, self).__init__()

    def forward(self, x, mask, label_encode=None):
        return x