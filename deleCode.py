# 废弃代码

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, context_dim, out_sample = 'none-3', in_sample = 'upsample'):
#         """
#         解码器模块
#         :param in_channels: 输入通道数（b*in_channels*h*w）
#         :param out_channels: 输出通道数（b*out_channels*h*w）
#         :param context_dim: 上下文维度（b*q*context_dim）
#         :param out_sample: 输出的采样方式
#         :param in_sample: 输入的采样方式，'none' or 'upsample'
#         """
#         super(DecoderBlock, self).__init__()
#         in_channels = in_channels * 3
#         self.block = PCAttWithSkipBlock(in_channels, out_channels, context_dim=context_dim)
#
#         if in_sample == 'none':
#             self.in_sample = None
#         else:
#             self.in_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#     def forward(self, x, mask, texture_encode, structure_encode, mask_t_e, mask_s_e, label_encode = None):
#
#         if self.in_sample is not None:
#             x = self.in_sample(x)
#             mask = self.in_sample(mask)
#         x = torch.cat([x, texture_encode, structure_encode], dim=1)
#         mask = torch.cat([mask, mask_t_e, mask_s_e], dim=1)
#
#         x, mask = self.block(x, mask, label_encode)
#
#         return x, mask


# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, context_dim=3):
#         super(Decoder, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # 用于将x_encode的通道数减少到out_channels
#         block_num = [64, 128, 256, 512, 512]
#         assert block_num[-1] == in_channels
#
#         block = DecoderBlock
#
#         self.block_1 = PCAttWithSkipBlock(in_channels*2, block_num[3], context_dim=context_dim)
#
#         self.block_2 = block(block_num[3]*2, block_num[2], context_dim, out_sample='none', in_sample='none')
#
#         self.block_3 = block(block_num[2]*2, block_num[1], context_dim, out_sample='none', in_sample='upsample')
#
#         self.block_4 = block(block_num[1]*2, block_num[0], context_dim, out_sample='none', in_sample='upsample')
#
#         self.block_5 = block(block_num[0]*2, out_channels, context_dim, out_sample='none', in_sample='upsample')
#
#     def forward(self, x, mask, texture_encode_list, structure_encode_list, mask_t_e_list, mask_s_e_list, context_encode = None):
#
#         texture_1, texture_2, texture_3, texture_4, texture_5 = texture_encode_list
#         structure_1, structure_2, structure_3, structure_4, structure_5 = structure_encode_list
#
#         mask_t_1, mask_t_2, mask_t_3, mask_t_4, mask_t_5 = mask_t_e_list
#         mask_s_1, mask_s_2, mask_s_3, mask_s_4, mask_s_5 = mask_s_e_list
#
#         x, mask = self.block_1(x, mask, context_encode)
#
#         x, mask = self.block_2(x, mask, texture_4, structure_4, mask_t_4, mask_s_4, context_encode)
#
#         x, mask = self.block_3(x, mask, texture_3, structure_3, mask_t_3, mask_s_3, context_encode)
#
#         x, mask = self.block_4(x, mask, texture_2, structure_2, mask_t_2, mask_s_2, context_encode)
#
#         x, mask = self.block_5(x, mask, texture_1, structure_1, mask_t_1, mask_s_1, context_encode)
#
#         return x