import torch
from torch import nn
import torch.nn.functional as F
from model.OtherModel import PCAttWithSkipBlock, PCWithSkipBlock, PCAttBlock, PCBlock
from myUtils import print_model_layers


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim = None, use_context = True, use_skip = False, in_sample = 'upsample', norm_type='instance', out_activ = 'relu'):
        """
        解码器模块
        :param in_channels: 输入通道数（b*in_channels*h*w）
        :param out_channels: 输出通道数（b*out_channels*h*w）
        :param context_dim: 上下文维度（b*q*context_dim）
        :param in_sample: 输入的采样方式，'none' or 'upsample'
        """
        super(DecoderBlock, self).__init__()
        self.use_context = use_context
        if use_context:
            if use_skip:
                self.block = PCAttWithSkipBlock(in_channels, out_channels, norm_type=norm_type, context_dim=context_dim, out_activ=out_activ)
            else:
                self.block = PCAttBlock(in_channels, out_channels, norm_type=norm_type, context_dim=context_dim)
        else:
            if use_skip:
                self.block = PCWithSkipBlock(in_channels, out_channels, norm_type=norm_type, out_activ=out_activ)
            else:
                self.block = PCBlock(in_channels, out_channels, norm_type=norm_type, out_activ=out_activ)

        if in_sample == 'none':
            self.in_sample = None
        else:
            self.in_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mask, fusion_encode, mask_f, context_encode = None):

        if self.in_sample is not None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        # print('x:', x.shape)
        # print('fusion_encode:', fusion_encode.shape)
        x = torch.cat([x, fusion_encode], dim=1)
        mask = torch.cat([mask, mask_f], dim=1)

        if self.use_context:
            x, mask = self.block(x, mask, context_encode)
        else:
            x, mask = self.block(x, mask)

        return x, mask

class Decoder(nn.Module):
    def __init__(self, out_channels, context_dim=3, block_num = [64, 128, 256, 512, 512], use_pconv = True):
        super(Decoder, self).__init__()
        self.in_channels = block_num[-1]*2
        self.out_channels = out_channels
        self.use_pconv = use_pconv

        block = DecoderBlock

        skip_channels = 3

        self.block_1 = PCAttBlock(self.in_channels, block_num[3], context_dim=context_dim)

        self.block_2 = block(block_num[3]*skip_channels, block_num[2], context_dim, in_sample='upsample')

        self.block_3 = block(block_num[2]*skip_channels, block_num[1], context_dim, in_sample='upsample')

        self.block_4 = block(block_num[1]*skip_channels, block_num[0], context_dim, in_sample='upsample')

        self.block_5 = block(block_num[0]*skip_channels, block_num[0], use_context=False, use_skip=False)

        self.out = nn.Sequential(
            nn.Conv2d(block_num[0], out_channels, 3, 1, 1),
            # nn.Tanh()
        )

    def forward(self, x, mask, fusion_list, mask_f_list, context_encode = None):

        fusion_1, fusion_2, fusion_3, fusion_4, fusion_5 = fusion_list
        mask_f_1, mask_f_2, mask_f_3, mask_f_4, mask_f_5 = mask_f_list

        x, mask = self.block_1(x, mask, context_encode)

        x, mask = self.block_2(x, mask, fusion_4, mask_f_4, context_encode)

        x, mask = self.block_3(x, mask, fusion_3, mask_f_3, context_encode)

        x, mask = self.block_4(x, mask, fusion_2, mask_f_2, context_encode)
        # 最后一个模块不用context_encode
        x, mask = self.block_5(x, mask, fusion_1, mask_f_1)

        x = self.out(x)

        # 限制到[-1, 1]
        x = torch.clamp(x, -1, 1)
        return x



if __name__ == '__main__':

    decoder = Decoder(512, 3)


