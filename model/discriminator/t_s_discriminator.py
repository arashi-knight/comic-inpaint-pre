import torch
import torch.nn as nn

from model.discriminator.structure_branch import EdgeDetector, StructureBranch
from model.discriminator.texture_branch import TextureBranch


class T_S_Discriminator(nn.Module):

    def __init__(self, image_in_channels, edge_in_channels):
        super(T_S_Discriminator, self).__init__()

        self.texture_branch = TextureBranch(in_channels=image_in_channels)
        self.structure_branch = StructureBranch(in_channels=edge_in_channels)

    def forward(self, output, edge):

        texture_pred = self.texture_branch(output)
        structure_pred = self.structure_branch(edge)

        return torch.cat((texture_pred, structure_pred), dim=1)
        

