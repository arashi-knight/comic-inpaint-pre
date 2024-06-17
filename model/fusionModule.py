import torch
from torch import nn

from myUtils import print_parameters


class BiGFF(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''

    def __init__(self, in_channels, out_channels):
        super(BiGFF, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, texture_feature, structure_feature):

        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)

        return torch.cat((texture_feature, structure_feature), dim=1)

class STFModule_hw(nn.Module):
    def __init__(self, in_channels):
        """
        门控特征融合模块
        :param in_channels: 输入特征的通道数
        """
        super(STFModule_hw, self).__init__()

        # TODO: 门控特征融合模块的实现



    def forward(self, texture_feature, structure_feature):
        # TODO: 门控特征融合模块的实现
        # 改成拼接
        fusion_feature = torch.cat((texture_feature, structure_feature), dim=1)

        return texture_feature, structure_feature, fusion_feature

if __name__ == "__main__":
    fusion = STFModule(512)
    sagate = SAGate(512, 512)
    bigff = BiGFF(512, 512)

    # texture_feature = torch.randn(1, 512, 224, 224)
    #
    # structure_feature = torch.randn(1, 512, 224, 224)
    #
    # x,y,z = fusion(texture_feature, structure_feature)
    #
    # print('texture_feature:', x.shape, 'structure_feature:', y.shape, 'fusion_feature:', z.shape)
    print('fusion:')
    print_parameters(fusion)
    print('sagate:')
    print_parameters(sagate)
    print('bigff:')
    print_parameters(bigff)