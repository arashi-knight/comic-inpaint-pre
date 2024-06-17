import torch
from torch import nn
import torch.nn.functional as F
from model.OtherModel import PCAttWithSkipBlock, PCAttBlock, PCWithSkipBlock, PCBlock, ConvAttWithSkipBlock, \
    ConvAttBlock
from model.fusionModule import STFModule, STFModuleConcat
from model.textureModule import HistModule
from myModel.fusionModule import STFModule_hw
from myUtils import print_model_layers, mean_tensors, make_f_mask_list


class StructureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=3, use_context = True, use_skip = False, use_pconv = True, out_sample='none-3', out_activ='relu'):
        super(StructureEncoder, self).__init__()
        self.use_context = use_context
        if use_context:
            if use_skip:
                if use_pconv:
                    self.block = PCAttWithSkipBlock(in_channels, out_channels, context_dim=context_channels, out_activ=out_activ, out_sample=out_sample)
                else:
                    self.block = ConvAttWithSkipBlock(in_channels, out_channels, context_dim=context_channels, out_activ=out_activ, out_sample=out_sample)
            else:
                if use_pconv:
                    self.block = PCAttBlock(in_channels, out_channels, context_dim=context_channels, out_sample=out_sample)
                else:
                    self.block = ConvAttBlock(in_channels, out_channels, context_dim=context_channels, out_sample=out_sample)


    def forward(self, x, mask, context_encode=None):
        if self.use_context:
            x_1, mask_1 = self.block(x, mask, context_encode)
        else:
            x_1, mask_1 = self.block(x, mask)
        return x_1, mask_1

class TextureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=3, use_context = True, use_skip = False, use_t_encode = True, out_sample='none-1', out_activ='relu'):
        super(TextureEncoder, self).__init__()
        if use_t_encode:
            self.textureblock = HistModule(in_channels, 3, padding=1)
        self.texture_activ = nn.ReLU(True)
        self.use_context = use_context
        # self.use_t_encode = use_t_encode
        if use_context:
            if use_skip:
                self.block = PCAttWithSkipBlock(in_channels, out_channels, context_dim=context_channels, out_activ=out_activ,out_sample=out_sample)
            else:
                self.block = PCAttBlock(in_channels, out_channels, context_dim=context_channels,out_sample=out_sample)
        else:
            if use_skip:
                self.block = PCWithSkipBlock(in_channels, out_channels, out_activ=out_activ,out_sample=out_sample)
            else:
                self.block = PCBlock(in_channels, out_channels,out_sample=out_sample, out_activ=out_activ)

    def forward(self, x, mask, context_encode=None):
        if hasattr(self, 'textureblock'):
            texture_encode = self.textureblock(x)
            x = x + texture_encode
            x = self.texture_activ(x)
        if self.use_context:
            x_1, mask_1 = self.block(x, mask, context_encode)
        else:
            x_1, mask_1 = self.block(x, mask)
        # print('texture:', x_1.shape, 'mask:', mask_1.shape)

        return x_1, mask_1


class TextureEncoder_conv(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=3, use_context = True, use_skip = False, use_t_encode = True, out_sample='none-1', out_activ='relu'):
        super(TextureEncoder_conv, self).__init__()
        if use_t_encode:
            self.textureblock = HistModule(in_channels, 3, padding=1)
        self.texture_activ = nn.ReLU(True)
        self.use_context = use_context
        # self.use_t_encode = use_t_encode
        if use_context:
            if use_skip:
                self.block = PCAttWithSkipBlock(in_channels, out_channels, context_dim=context_channels, out_activ=out_activ,out_sample=out_sample)
            else:
                self.block = PCAttBlock(in_channels, out_channels, context_dim=context_channels,out_sample=out_sample)
        else:
            if use_skip:
                self.block = PCWithSkipBlock(in_channels, out_channels, out_activ=out_activ,out_sample=out_sample)
            else:
                self.block = PCBlock(in_channels, out_channels,out_sample=out_sample, out_activ=out_activ)






class Encoder(nn.Module):
    def __init__(self, in_channels, context_channels=3, block_num = [64, 128, 256, 512, 512], use_pconv = True):
        super(Encoder, self).__init__()


        self.use_pconv = use_pconv

        s_block = StructureEncoder
        t_block = TextureEncoder
        f_block = STFModule

        self.t_block_1 = t_block(in_channels, block_num[0], context_channels)
        self.t_block_2 = t_block(block_num[0], block_num[1], context_channels, out_sample='down-3')
        self.t_block_3 = t_block(block_num[1], block_num[2], context_channels, out_sample='down-3')
        self.t_block_4 = t_block(block_num[2], block_num[3], context_channels, out_sample='down-3')
        self.t_block_5 = t_block(block_num[3], block_num[4], context_channels, out_sample='down-3')

        self.s_block_1 = s_block(in_channels, block_num[0], context_channels)
        self.s_block_2 = s_block(block_num[0], block_num[1], context_channels, out_sample='down-3')
        self.s_block_3 = s_block(block_num[1], block_num[2], context_channels, out_sample='down-3')
        self.s_block_4 = s_block(block_num[2], block_num[3], context_channels, out_sample='down-3')
        self.s_block_5 = s_block(block_num[3], block_num[4], context_channels, out_sample='down-3')

        self.f_block_1 = f_block(block_num[0])
        self.f_block_2 = f_block(block_num[1])
        self.f_block_3 = f_block(block_num[2])
        self.f_block_4 = f_block(block_num[3])
        self.f_block_5 = f_block(block_num[4])

    def forward(self, x, structure, mask, context_encode=None):

        texture, mask_t_1 = self.t_block_1(x, mask, context_encode)
        structure, mask_s_1 = self.s_block_1(structure, mask, context_encode)
        texture, structure, fusion_1 = self.f_block_1(texture, structure)

        texture, mask_t_2 = self.t_block_2(texture, mask_t_1, context_encode)
        structure, mask_s_2 = self.s_block_2(structure, mask_s_1, context_encode)
        texture, structure, fusion_2 = self.f_block_2(texture, structure)

        texture, mask_t_3 = self.t_block_3(texture, mask_t_2, context_encode)
        structure, mask_s_3 = self.s_block_3(structure, mask_s_2, context_encode)
        texture, structure, fusion_3 = self.f_block_3(texture, structure)

        texture, mask_t_4 = self.t_block_4(texture, mask_t_3, context_encode)
        structure, mask_s_4 = self.s_block_4(structure, mask_s_3, context_encode)
        texture, structure, fusion_4 = self.f_block_4(texture, structure)

        texture, mask_t_5 = self.t_block_5(texture, mask_t_4, context_encode)
        structure, mask_s_5 = self.s_block_5(structure, mask_s_4, context_encode)
        texture, structure, fusion_5 = self.f_block_5(texture, structure)

        mask_f_list = make_f_mask_list([mask_s_1, mask_s_2, mask_s_3, mask_s_4, mask_s_5],
                                       [mask_t_1, mask_t_2, mask_t_3, mask_t_4, mask_t_5],
                                       f_mode='cat')

        return [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5], mask_f_list

class Encoder_v2(nn.Module):
    def __init__(self, in_channels, context_channels=3, block_num = [64, 128, 256, 512, 512], use_pconv = True):
        super(Encoder_v2, self).__init__()


        self.use_pconv = use_pconv

        s_block = StructureEncoder
        t_block = TextureEncoder
        f_block = STFModule

        self.t_block_1 = t_block(in_channels, block_num[0], context_channels)
        self.t_block_2 = t_block(block_num[0], block_num[1], context_channels)
        self.t_block_3 = t_block(block_num[1], block_num[2], context_channels)
        self.t_block_4 = t_block(block_num[2], block_num[3], context_channels)
        self.t_block_5 = t_block(block_num[3], block_num[4], context_channels)

        self.s_block_1 = s_block(in_channels, block_num[0], context_channels)
        self.s_block_2 = s_block(block_num[0], block_num[1], context_channels)
        self.s_block_3 = s_block(block_num[1], block_num[2], context_channels)
        self.s_block_4 = s_block(block_num[2], block_num[3], context_channels)
        self.s_block_5 = s_block(block_num[3], block_num[4], context_channels)

        self.f_block_1 = f_block(block_num[0])
        self.f_block_2 = f_block(block_num[1])
        self.f_block_3 = f_block(block_num[2])
        self.f_block_4 = f_block(block_num[3])
        self.f_block_5 = f_block(block_num[4])

    # 把输入的texture，structure, mask_t, mask_s过一个maxpool2d
    def pool(self, texture, structure, mask_t, mask_s):
        texture = F.max_pool2d(texture, 2)
        structure = F.max_pool2d(structure, 2)
        mask_t = F.max_pool2d(mask_t, 2)
        mask_s = F.max_pool2d(mask_s, 2)
        return texture, structure, mask_t, mask_s

    def forward(self, x, structure, mask, context_encode=None):

        texture, mask_t_1 = self.t_block_1(x, mask, context_encode)
        structure, mask_s_1 = self.s_block_1(structure, mask, context_encode)
        texture, structure, fusion_1 = self.f_block_1(texture, structure)

        texture, structure, mask_t_2, mask_s_2 = self.pool(texture, structure, mask_t_1, mask_s_1)
        texture, mask_t_2 = self.t_block_2(texture, mask_t_2, context_encode)
        structure, mask_s_2 = self.s_block_2(structure, mask_s_2, context_encode)
        texture, structure, fusion_2 = self.f_block_2(texture, structure)

        texture, structure, mask_t_3, mask_s_3 = self.pool(texture, structure, mask_t_2, mask_s_2)
        texture, mask_t_3 = self.t_block_3(texture, mask_t_3, context_encode)
        structure, mask_s_3 = self.s_block_3(structure, mask_s_3, context_encode)
        texture, structure, fusion_3 = self.f_block_3(texture, structure)

        texture, structure, mask_t_4, mask_s_4 = self.pool(texture, structure, mask_t_3, mask_s_3)
        texture, mask_t_4 = self.t_block_4(texture, mask_t_4, context_encode)
        structure, mask_s_4 = self.s_block_4(structure, mask_s_4, context_encode)
        texture, structure, fusion_4 = self.f_block_4(texture, structure)

        texture, structure, mask_t_5, mask_s_5 = self.pool(texture, structure, mask_t_4, mask_s_4)
        texture, mask_t_5 = self.t_block_5(texture, mask_t_5, context_encode)
        structure, mask_s_5 = self.s_block_5(structure, mask_s_5, context_encode)
        texture, structure, fusion_5 = self.f_block_5(texture, structure)

        # texture, structure, mask_t_2, mask_s_2 = self.pool(texture, structure, mask_t_2, mask_s_2)
        # texture, mask_t_3 = self.t_block_3(texture, mask_t_2, context_encode)
        # structure, mask_s_3 = self.s_block_3(structure, mask_s_2, context_encode)
        # texture, structure, fusion_3 = self.f_block_3(texture, structure)
        #
        # texture, structure, mask_t_3, mask_s_3 = self.pool(texture, structure, mask_t_3, mask_s_3)
        # texture, mask_t_4 = self.t_block_4(texture, mask_t_3, context_encode)
        # structure, mask_s_4 = self.s_block_4(structure, mask_s_3, context_encode)
        # texture, structure, fusion_4 = self.f_block_4(texture, structure)
        #
        # texture, structure, mask_t_4, mask_s_4 = self.pool(texture, structure, mask_t_4, mask_s_4)
        # texture, mask_t_5 = self.t_block_5(texture, mask_t_4, context_encode)
        # structure, mask_s_5 = self.s_block_5(structure, mask_s_4, context_encode)
        # texture, structure, fusion_5 = self.f_block_5(texture, structure)


        # return [texture_1, texture_2, texture_3, texture_4, texture_5], [mask_t_1, mask_t_2, mask_t_3, mask_t_4, mask_t_5], \
        #           [structure_1, structure_2, structure_3, structure_4, structure_5], [mask_s_1, mask_s_2, mask_s_3, mask_s_4, mask_s_5], \
        #             [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5]

        mask_f_list = make_f_mask_list([mask_s_1, mask_s_2, mask_s_3, mask_s_4, mask_s_5],
                                       [mask_t_1, mask_t_2, mask_t_3, mask_t_4, mask_t_5],
                                       f_mode='cat')


        return [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5], mask_f_list


class Encoder_v3(nn.Module):
    def __init__(self, in_channels, context_channels=3, block_num = [64, 128, 256, 512, 512], use_pconv = True):
        super(Encoder_v3, self).__init__()


        self.use_pconv = use_pconv

        s_block = StructureEncoder
        t_block = TextureEncoder
        f_block = STFModule_hw

        self.t_block_1 = t_block(in_channels, block_num[0], context_channels)
        self.t_block_2 = t_block(block_num[0], block_num[1], context_channels)
        self.t_block_3 = t_block(block_num[1], block_num[2], context_channels)
        self.t_block_4 = t_block(block_num[2], block_num[3], context_channels)
        self.t_block_5 = t_block(block_num[3], block_num[4], context_channels)

        self.s_block_1 = s_block(in_channels, block_num[0], context_channels)
        self.s_block_2 = s_block(block_num[0], block_num[1], context_channels)
        self.s_block_3 = s_block(block_num[1], block_num[2], context_channels)
        self.s_block_4 = s_block(block_num[2], block_num[3], context_channels)
        self.s_block_5 = s_block(block_num[3], block_num[4], context_channels)

        self.f_block_1 = f_block(block_num[0])
        self.f_block_2 = f_block(block_num[1])
        self.f_block_3 = f_block(block_num[2])
        self.f_block_4 = f_block(block_num[3])
        self.f_block_5 = f_block(block_num[4])

    # 把输入的texture，structure, mask_t, mask_s过一个maxpool2d
    def pool(self, texture, structure, mask_t, mask_s):
        texture = F.max_pool2d(texture, 2)
        structure = F.max_pool2d(structure, 2)
        mask_t = F.max_pool2d(mask_t, 2)
        mask_s = F.max_pool2d(mask_s, 2)
        return texture, structure, mask_t, mask_s

    def forward(self, x, structure, mask, context_encode=None):

        texture, mask_t_1 = self.t_block_1(x, mask, context_encode)
        structure, mask_s_1 = self.s_block_1(structure, mask, context_encode)
        texture, structure, fusion_1 = self.f_block_1(texture, structure)

        texture, structure, mask_t_2, mask_s_2 = self.pool(texture, structure, mask_t_1, mask_s_1)
        texture, mask_t_2 = self.t_block_2(texture, mask_t_2, context_encode)
        structure, mask_s_2 = self.s_block_2(structure, mask_s_2, context_encode)
        texture, structure, fusion_2 = self.f_block_2(texture, structure)

        texture, structure, mask_t_3, mask_s_3 = self.pool(texture, structure, mask_t_2, mask_s_2)
        texture, mask_t_3 = self.t_block_3(texture, mask_t_3, context_encode)
        structure, mask_s_3 = self.s_block_3(structure, mask_s_3, context_encode)
        texture, structure, fusion_3 = self.f_block_3(texture, structure)

        texture, structure, mask_t_4, mask_s_4 = self.pool(texture, structure, mask_t_3, mask_s_3)
        texture, mask_t_4 = self.t_block_4(texture, mask_t_4, context_encode)
        structure, mask_s_4 = self.s_block_4(structure, mask_s_4, context_encode)
        texture, structure, fusion_4 = self.f_block_4(texture, structure)

        texture, structure, mask_t_5, mask_s_5 = self.pool(texture, structure, mask_t_4, mask_s_4)
        texture, mask_t_5 = self.t_block_5(texture, mask_t_5, context_encode)
        structure, mask_s_5 = self.s_block_5(structure, mask_s_5, context_encode)
        texture, structure, fusion_5 = self.f_block_5(texture, structure)

        mask_f_list = make_f_mask_list([mask_s_1, mask_s_2, mask_s_3, mask_s_4, mask_s_5],
                                       [mask_t_1, mask_t_2, mask_t_3, mask_t_4, mask_t_5],
                                       f_mode='cat')


        return [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5], mask_f_list








if __name__ == '__main__':

    encoder = Encoder(3)
    x = torch.randn(3, 3, 256, 256)
    structure = torch.randn(3, 3, 256, 256)
    mask = torch.randn(3, 3, 256, 256)
    context_encode = torch.randn(3, 1, 3)

    # print_model_layers(encoder, x = x, structure = structure, mask = mask, context_encode = context_encode)
    fusion_list, masks_list = encoder(x, structure, mask, context_encode)

    print('fusion_list:', [i.shape for i in fusion_list])
    print('masks_list:', [i.shape for i in masks_list])

    encoder_v2 = Encoder_v2(3)

    fusion_list_v2, masks_list_v2 = encoder_v2(x, structure, mask, context_encode)

    print('fusion_list_v2:', [i.shape for i in fusion_list_v2])
    print('masks_list_v2:', [i.shape for i in masks_list_v2])

    # print('texture_list:', [i.shape for i in texture_list])
