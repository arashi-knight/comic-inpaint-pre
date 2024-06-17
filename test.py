import sys
from time import sleep

import torch
from colorama import Fore
from torch import nn
from tqdm import tqdm

from config import Config
from model.OtherModel import CrossAttention, PCAttWithSkipBlock


class testmodel(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(testmodel, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)

        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)

        self.conv_3 = nn.Conv2d(input_channels*3, output_channels, 3, padding=1)

        self.conv_4 = nn.Conv2d(input_channels*2, output_channels, 3, padding=1)


    def forward(self, x):
        return x

# 判断一个tensor是否只包含0和1
def is_binary_tensor(tensor):
    return torch.all(torch.logical_or(tensor == 0, tensor == 1))

if __name__ == '__main__':

    config = Config()

    # classes, train_dataloader, test_dataloader, val_dataloader = get_dataloader(config)

