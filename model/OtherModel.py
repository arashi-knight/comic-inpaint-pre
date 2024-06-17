import re

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Callable, Type, TypeVar
import torch.nn.functional as F
from model.ModelUtils import default, exists, weights_init
from myUtils import get_activation, NormWrapper


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, is_spectral_norm = False):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)

        if is_spectral_norm:
            self.input_conv = nn.utils.spectral_norm(self.input_conv)
        self.input_conv.apply(weights_init('kaiming'))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.mask_weight = torch.ones(out_channels, in_channels, kernel_size, kernel_size)

        # torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        # 不更新mask参数
        # for param in self.mask_conv.parameters():
        #     param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        x = input * mask

        # print("x: ", x.shape)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            if self.mask_weight.type() != input.type():
                self.mask_weight = self.mask_weight.to(input)
            output_mask = F.conv2d(mask, self.mask_weight, bias=None, stride=self.stride,
                                      padding=self.padding, dilation=self.dilation, groups=self.groups)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='instance', sample='none-3', activ='relu', conv_bias=False, is_spectral_norm = True):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias, is_spectral_norm=is_spectral_norm)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias, is_spectral_norm=is_spectral_norm)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias, is_spectral_norm=is_spectral_norm)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias, is_spectral_norm=is_spectral_norm)


        self.norm = NormWrapper(out_ch, norm_type)
        if activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activ == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, mask):
        h, h_mask = self.conv(x, mask)
        h = self.norm(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class CBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='instance', sample='none-3', activ='relu', conv_bias=False, is_spectral_norm = True):
        super().__init__()
        if sample == 'down-5':
            self.conv = nn.Conv2d(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = nn.Conv2d(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        elif sample == 'none-1':
            self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=conv_bias)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if is_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.norm = NormWrapper(out_ch, norm_type)
        # 获取激活函数
        self.activation = get_activation(activ)

    def forward(self, input):
        h = self.conv(input)
        h = self.norm(h)
        h = self.activation(h)
        return h

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        # print("context: ", context.shape)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BNRelu(nn.Module):
    """
    Batch Normalization + ReLU
    """
    def __init__(self, in_channels):
        super(BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x

class InplaceReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)

class MobileBlock(torch.nn.Module):
    """
    MobileNet-style base block
    MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381
    Pixel-wise (shuffle) -> Depth-wise -> Pixel-wise
    """

    def __init__(self,
                 input_size: int,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expansion: int = 1,
                 kernel: int = 3,
                 groups: int = 1,
                 batch_norm_2d: Type[torch.nn.BatchNorm2d] = torch.nn.BatchNorm2d,
                 relu: Callable[[], nn.Module] = InplaceReLU,
                 residual: bool = False
                 ):
        super().__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion = expansion
        self.kernel = kernel
        self.groups = groups
        self.residual = residual and stride == 1 and in_channels == out_channels

        inner_channels = in_channels * expansion
        self.block = nn.Sequential(
            # pixel wise
            nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, groups=groups, bias=False),
            batch_norm_2d(num_features=inner_channels),
            relu(),
            ShuffleBlock(groups=groups) if groups > 1 else nn.Sequential(),
            # depth wise
            nn.Conv2d(in_channels=inner_channels, out_channels=inner_channels, groups=inner_channels,
                      kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
            batch_norm_2d(num_features=inner_channels),
            relu(),
            # pixel wise
            nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, groups=groups, bias=False),
            batch_norm_2d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return self.block(x) + x
        else:
            return self.block(x)

class ShuffleBlock(nn.Module):
    """
    shuffle channels
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, https://arxiv.org/abs/1707.01083
    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel shuffle: [n, c, h, w] -> [n, g, c/g, h, w] -> [n, c/g, g, h, w] -> [n, c, h, w]
        n, c, h, w = x.size()
        g = self.groups
        return x.view(n, g, c // g, h, w).transpose(1, 2).contiguous().view(n, c, h, w)

class LayerNormWrapper(nn.Module):
    def __init__(self, num_features):
        super(LayerNormWrapper, self).__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]], elementwise_affine=False).cuda()(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            LayerNormWrapper(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            LayerNormWrapper(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class PCAttWithSkipBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, hidden_scale = 4, out_sample='none-3', norm_type = 'instance', out_activ ='relu'):
        super().__init__()

        hidden_dim = in_dim // hidden_scale if in_dim // hidden_scale > 8 else 8

        self.con1 = PCBActiv(in_dim, hidden_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.att = CrossAttention(hidden_dim, context_dim, heads=8, dim_head=64)

        self.con2 = PCBActiv(hidden_dim, out_dim, norm_type=norm_type, sample=out_sample, activ='relu', conv_bias=False)

        if in_dim != out_dim or 'none' not in out_sample:
            self.downsample = CBActiv(in_dim, out_dim, norm_type='none', sample=out_sample, activ='none', conv_bias=False)

        self.out_activ = get_activation(out_activ)


    def forward(self, x, mask, context=None):
        x_in = x
        # context = default(context, x)
        # print('pcattwithskipblock')
        # print("x: ", x.shape)
        # print("mask: ", mask.shape)
        x, mask = self.con1(x, mask)

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.att(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        out_x, out_mask = self.con2(x, mask)

        if hasattr(self, 'downsample'):
            out_x = out_x + self.downsample(x_in)
        else:
            out_x = out_x + x_in


        out_x = self.out_activ(out_x)

        return out_x, out_mask

class PCAttBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, hidden_scale = 4, out_sample='none-3', norm_type = 'instance', out_activ ='relu'):
        super().__init__()

        hidden_dim = in_dim // hidden_scale if in_dim // hidden_scale > 8 else 8

        self.con1 = PCBActiv(in_dim, hidden_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.att = CrossAttention(hidden_dim, context_dim, heads=8, dim_head=64)

        self.con2 = PCBActiv(hidden_dim, out_dim, norm_type=norm_type, sample=out_sample, activ=out_activ, conv_bias=False)

    def forward(self, x, mask, context=None):

        x, mask = self.con1(x, mask)

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.att(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        out_x, out_mask = self.con2(x, mask)

        return out_x, out_mask

class PCBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, out_sample='none-3', norm_type = 'instance', out_activ ='relu'):
        super().__init__()

        self.con1 = PCBActiv(in_dim, out_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.con2 = PCBActiv(out_dim, out_dim, norm_type=norm_type, sample=out_sample, activ=out_activ, conv_bias=False)

    def forward(self, x, mask):

        x, mask = self.con1(x, mask)

        out_x, out_mask = self.con2(x, mask)

        return out_x, out_mask

class PCWithSkipBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, out_sample='none-3', norm_type = 'instance', out_activ = 'relu'):
        super().__init__()

        self.con1 = PCBActiv(in_dim, out_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.con2 = PCBActiv(out_dim, out_dim, norm_type=norm_type, sample=out_sample, activ='relu', conv_bias=False)

        if in_dim != out_dim or 'none' not in out_sample:
            self.downsample = CBActiv(in_dim, out_dim, norm_type='none', sample=out_sample, activ='none', conv_bias=False)

        self.out_activ = get_activation(out_activ)

    def forward(self, x, mask):

        out_x, out_mask = self.con1(x, mask)

        out_x, out_mask = self.con2(out_x, out_mask)

        if hasattr(self, 'downsample'):
            out_x = out_x + self.downsample(x)
        else:
            out_x = out_x + x


        out_x = self.out_activ(out_x)

        return out_x, out_mask

class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, out_sample='none-3', norm_type = 'instance', out_activ = 'relu'):
        super().__init__()

        self.con1 = CBActiv(in_dim, out_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.con2 = CBActiv(out_dim, out_dim, norm_type=norm_type, sample=out_sample, activ=out_activ, conv_bias=False)

    def forward(self, x):

        x = self.con1(x)

        out_x = self.con2(x)

        return out_x

class ConvAttBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, out_sample='none-3', out_activ = 'relu', norm_type = 'instance'):
        super().__init__()

        hidden_dim = in_dim // 4 if in_dim // 4 > 8 else 8

        self.con1 = CBActiv(in_dim, hidden_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.att = CrossAttention(hidden_dim, context_dim, heads=8, dim_head=64)

        self.con2 = CBActiv(hidden_dim, out_dim, norm_type=norm_type, sample=out_sample, activ=out_activ, conv_bias=False)

    def forward(self, x, context=None):

        x = self.con1(x)

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.att(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        out_x = self.con2(x)

        return out_x

class ConvWithSkipBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, out_sample='none-3', norm_type = 'instance', out_activ = 'relu'):
        super().__init__()

        self.con1 = CBActiv(in_dim, out_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.con2 = CBActiv(out_dim, out_dim, norm_type=norm_type, sample=out_sample, activ='relu', conv_bias=False)

        if in_dim != out_dim or 'none' not in out_sample:
            self.downsample = CBActiv(in_dim, out_dim, norm_type='none', sample=out_sample, activ='none', conv_bias=False)

        self.out_activ = get_activation(out_activ)

    def forward(self, x):

        out_x = self.con1(x)

        out_x = self.con2(out_x)

        if hasattr(self, 'downsample'):
            out_x = out_x + self.downsample(x)
        else:
            out_x = out_x + x


        out_x = self.out_activ(out_x)

        return out_x

class ConvAttWithSkipBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, out_sample='none-3', out_activ = 'none', norm_type='instance'):
        super().__init__()

        hide_dim = in_dim // 4 if in_dim // 4 > 8 else 8

        self.con1 = CBActiv(in_dim, hide_dim, norm_type=norm_type, sample='none-3', activ='relu', conv_bias=False)

        self.att = CrossAttention(hide_dim, context_dim, heads=8, dim_head=64)

        self.con2 = CBActiv(hide_dim, out_dim, norm_type=norm_type, sample=out_sample, activ='relu', conv_bias=False)

        # out_sample 字段不包含none时，需要添加下采样层
        if in_dim != out_dim or 'none' not in out_sample:
            self.downsample = CBActiv(in_dim, out_dim, norm_type='none', sample=out_sample, activ='none', conv_bias=False)

        self.out_activ = get_activation(out_activ)

    def forward(self, x, context=None):
        x_in = x

        x = self.con1(x)

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.att(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        out_x = self.con2(x)

        if hasattr(self, 'downsample'):
            out_x = out_x + self.downsample(x_in)
        else:
            out_x = out_x + x_in


        out_x = self.out_activ(out_x)

        return out_x


