import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from model.OtherModel import CBActiv
from myUtils import print_parameters


class HistModule(nn.Module):
    '''Histogram Module'''

    def __init__(self,in_channels,kernel_size,dim=2,num_bins=1,
                 stride=1,padding=0,normalize_count=False,normalize_bins = True,
                 count_include_pad=False,
                 ceil_mode=False):
        super(HistModule, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode



        self.bin_centers_conv = nn.Conv2d(self.in_channels, self.in_channels, 1,
                                          groups=self.in_channels, bias=True)

        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_widths_conv = nn.Conv2d(self.in_channels,
                                         self.in_channels, 1,
                                         groups=self.in_channels,
                                         bias=True)
        self.bin_widths_conv.bias.data.fill_(1)
        # self.bin_widths_conv.bias.requires_grad = False
        self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                      padding=self.padding, ceil_mode=self.ceil_mode,
                                      count_include_pad=self.count_include_pad)


    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers: |x-center|
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        xx = torch.abs(self.bin_centers_conv(xx))

        # Pass through second convolution to learn bin widths 1-w*|x-center|
        self.bin_widths_conv.bias.data.fill_(1)
        self.bin_widths_conv.bias.requires_grad = False
        xx = self.bin_widths_conv(xx)

        # Pass through relu
        xx = F.relu(xx)

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins

        # Image Data
        n, c, h, w = xx.size()
        xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
        # print('xx_sum:', xx_sum.shape)
        xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
        # print('xx_sum:', xx_sum.shape)
        xx = xx / xx_sum
        # print('xx:', xx.shape)

        return xx

class MHTModule(nn.Module):
    '''Multi-Histogram Texture Module'''

    def __init__(self, in_channels, out_channels, num_bins_list = [1,4,8], stride=1):
        super(MHTModule, self).__init__()

        # TODO：多尺度直方图模块

    def forward(self, x):
        # TODO：多尺度直方图模块

        return x


if __name__ == '__main__':
    # test HistModule
    # hist_module = HistModule_v2(512, 3, padding=1, num_bins=4)
    # input = torch.randn(1, 512, 16, 16)
    # # output = hist_module(input)
    # # # print('output:', output.shape)
    # # # print_parameters(hist_module)

    mh_encoder = MHTModule(512, 512, 3)
    input = torch.randn(1, 512, 16, 16)
    output = mh_encoder(input)
    print('output:', output.shape)
    print_parameters(mh_encoder)