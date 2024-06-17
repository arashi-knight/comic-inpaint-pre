import torch
from torch import nn
import torch.nn.functional as F

from model.OtherModel import ConvWithSkipBlock, ConvAttBlock, ConvAttWithSkipBlock, ConvBlock, ResnetBlock
from model.fusionModule import STFModule_hw
from model.textureModule import HistModule, MHTModule
from myUtils import print_parameters, print_trainable_parameters, print_parameters_by_layer_name
from svae.svae import ScreenVAE


class TextureEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=None, use_context = False, use_skip = True, use_texture = True):
        super(TextureEncoderBlock, self).__init__()
        self.use_context = use_context
        if use_texture:
            # self.textureblock = HistModule(in_channels, 3, padding=1)
            self.textureblock = MHTModule(in_channels, 3)
        # self.textureblock = nn.Identity()
        if use_context:
            if use_skip:
                self.block = ConvAttWithSkipBlock(in_channels, out_channels, context_channels)
            else:
                self.block = ConvAttBlock(in_channels, out_channels, context_channels)
        else:
            if use_skip:
                self.block = ConvWithSkipBlock(in_channels, out_channels)
            else:
                self.block = ConvBlock(in_channels, out_channels)

        # self.activation = nn.ReLU(True)

    def forward(self, x, context_encode = None):
        if hasattr(self, 'textureblock'):
            texture_encode = self.textureblock(x)
            x = x + texture_encode
        if self.use_context:
            return self.block(x, context_encode)
        else:
            return self.block(x)

class StructureEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=None, use_context = False, use_skip = True):
        super(StructureEncoderBlock, self).__init__()
        self.use_context = use_context
        if use_context:
            if use_skip:
                self.block = ConvAttWithSkipBlock(in_channels, out_channels, context_channels)
            else:
                self.block = ConvAttBlock(in_channels, out_channels, context_channels)
        else:
            if use_skip:
                self.block = ConvWithSkipBlock(in_channels, out_channels)
            else:
                self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x, context_encode = None):
        if self.use_context:
            return self.block(x, context_encode)
        else:
            return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels = None, use_context = False, use_skip = True, norm_type='instance', out_activ = 'relu'):
        """
        解码器模块
        :param in_channels: 输入通道数（b*in_channels*h*w）
        :param out_channels: 输出通道数（b*out_channels*h*w）
        :param context_channels: 上下文维度（b*q*context_channels）
        :param in_sample: 输入的采样方式，'none' or 'upsample'
        """
        super(DecoderBlock, self).__init__()
        self.use_context = use_context
        if use_context:
            if use_skip:
                self.block = ConvAttWithSkipBlock(in_channels, out_channels, norm_type=norm_type, context_dim=context_channels, out_activ=out_activ)
            else:
                self.block = ConvAttBlock(in_channels, out_channels, norm_type=norm_type, context_dim=context_channels, out_activ=out_activ)
        else:
            if use_skip:
                self.block = ConvWithSkipBlock(in_channels, out_channels, norm_type=norm_type, out_activ=out_activ)
            else:
                self.block = ConvBlock(in_channels, out_channels, norm_type=norm_type, out_activ=out_activ)

    def forward(self, x, context_encode = None):
        if self.use_context:
            x = self.block(x, context_encode)
        else:
            x = self.block(x)

        return x

class ComicNet_v2(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels=3, base_block_num = 32):
        super(ComicNet_v2, self).__init__()

        self.svae = ScreenVAE().cuda()

        block_num = [base_block_num, base_block_num*2, base_block_num*4, base_block_num*8, base_block_num*8]


        t_block = TextureEncoderBlock
        s_block = StructureEncoderBlock
        f_block = STFModule_hw
        d_block = DecoderBlock
        d_block_scale = 3

        # texture encoder(输入为image+mask)
        self.t_encoder_1 = t_block(2, block_num[0],context_channels=context_channels)
        self.t_encoder_2 = t_block(block_num[0], block_num[1],context_channels=context_channels)
        self.t_encoder_3 = t_block(block_num[1], block_num[2],context_channels=context_channels)
        self.t_encoder_4 = t_block(block_num[2], block_num[3],context_channels=context_channels)
        self.t_encoder_5 = t_block(block_num[3], block_num[4],context_channels=context_channels)

        # structure encoder(输入为line+svae+mask)
        self.s_encoder_1 = s_block(6, block_num[0],context_channels=context_channels)
        self.s_encoder_2 = s_block(block_num[0], block_num[1],context_channels=context_channels)
        self.s_encoder_3 = s_block(block_num[1], block_num[2],context_channels=context_channels)
        self.s_encoder_4 = s_block(block_num[2], block_num[3],context_channels=context_channels)
        self.s_encoder_5 = s_block(block_num[3], block_num[4],context_channels=context_channels)

        # fusion module
        self.fusion_1 = f_block(block_num[0])
        self.fusion_2 = f_block(block_num[1])
        self.fusion_3 = f_block(block_num[2])
        self.fusion_4 = f_block(block_num[3])
        self.fusion_5 = f_block(block_num[4])

        # middle layer
        self.middle = nn.Sequential(
            nn.Conv2d(block_num[-1] * 2, block_num[-1], 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        for _ in range(6):
            block = ResnetBlock(block_num[-1], 2)
            self.middle.add_module('block_%d' % _, block)

        # decoder
        self.decoder_1 = d_block(block_num[4], block_num[3],context_channels=context_channels)
        self.decoder_2 = d_block(block_num[3]*d_block_scale, block_num[2],context_channels=context_channels)
        self.decoder_3 = d_block(block_num[2]*d_block_scale, block_num[1],context_channels=context_channels)
        self.decoder_4 = d_block(block_num[1]*d_block_scale, block_num[0],context_channels=context_channels)
        self.decoder_5 = d_block(block_num[0]*d_block_scale, block_num[0],context_channels=context_channels)

        self.out = nn.Sequential(
            nn.Conv2d(block_num[0], out_channels, 3, 1, 1),
            nn.Tanh()
        )

        self.init_weights()



    def forward(self, image, structure, mask, context_encode = None):

        svae = self.svae(image, structure,rep = True)
        image = torch.cat([image, mask], dim=1)
        structure = torch.cat([structure, svae, mask], dim=1)

        # encoder_1
        texture = self.t_encoder_1(image, context_encode)
        structure = self.s_encoder_1(structure, context_encode)
        # fusion module
        texture, structure, fusion_1 = self.fusion_1(texture, structure)
        # encoder_2
        texture = self.t_encoder_2(F.max_pool2d(texture, 2), context_encode)
        structure = self.s_encoder_2(F.max_pool2d(structure, 2), context_encode)
        # fusion module
        texture, structure, fusion_2 = self.fusion_2(texture, structure)
        # encoder_3
        texture = self.t_encoder_3(F.max_pool2d(texture, 2), context_encode)
        structure = self.s_encoder_3(F.max_pool2d(structure, 2), context_encode)
        # fusion module
        texture, structure, fusion_3 = self.fusion_3(texture, structure)
        # encoder_4
        texture = self.t_encoder_4(F.max_pool2d(texture, 2), context_encode)
        structure = self.s_encoder_4(F.max_pool2d(structure, 2), context_encode)
        # fusion module
        texture, structure, fusion_4 = self.fusion_4(texture, structure)
        # encoder_5
        texture = self.t_encoder_5(F.max_pool2d(texture, 2), context_encode)
        structure = self.s_encoder_5(F.max_pool2d(structure, 2), context_encode)
        # fusion module
        texture, structure, fusion_5 = self.fusion_5(texture, structure)

        # middle layer
        middle = self.middle(fusion_5)

        # decoder_1
        output = self.decoder_1(middle, context_encode)
        # decoder_2
        output = F.interpolate(output, scale_factor=2, mode='bilinear')
        # print('output:', output.shape)
        output = torch.cat([output, fusion_4], dim=1)
        output = self.decoder_2(output, context_encode)
        # decoder_3
        output = F.interpolate(output, scale_factor=2, mode='bilinear')
        output = torch.cat([output, fusion_3], dim=1)
        output = self.decoder_3(output, context_encode)
        # decoder_4
        output = F.interpolate(output, scale_factor=2, mode='bilinear')
        output = torch.cat([output, fusion_2], dim=1)
        output = self.decoder_4(output, context_encode)
        # decoder_5
        output = F.interpolate(output, scale_factor=2, mode='bilinear')
        output = torch.cat([output, fusion_1], dim=1)
        output = self.decoder_5(output, context_encode)
        output = self.out(output)
        # 限制输出范围-1,1
        output = torch.clamp(output, -1, 1)
        return output

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

if __name__ == '__main__':
    model = ComicNet_v2(1, 1, 3, 32).cuda()

    image_path = 'test_image/image.jpg'
    structure_path = 'test_image/structure.jpg'
    mask_path = 'test_image/mask.jpg'

    x = torch.randn(1, 1, 256, 256).cuda()
    structure = torch.randn(1, 1, 256, 256).cuda()
    mask = torch.randn(1, 1, 256, 256).cuda()

    context_encode = torch.randn(1, 77, 3).cuda()

    out = model(x, structure, mask, context_encode)

    print(out.shape)
    print_trainable_parameters(model)

    # print_trainable_parameters(model)

    print_parameters_by_layer_name(model, 'encoder')
    print_parameters_by_layer_name(model, 'fusion')
    print_parameters_by_layer_name(model, 'decoder')




    # print_model_layers(model, x = x, structure = structure, mask = mask, context_encode = context_encode)