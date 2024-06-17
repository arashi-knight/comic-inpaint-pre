import copy
import math
import sys

import lpips
import numpy as np
import pytorch_fid
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_fid import fid_score
from torchvision.models import VGG19_Weights, VGG16_Weights
from torchvision.transforms import transforms

from model.core.spectral_norm import use_spectral_norm
from myUtils import dilate_mask, shrunk_mask, distance_transform, show_tensor
from tools.ssim import MS_SSIM


def get_vgg19(device):
    cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            layer.padding_mode = 'reflect'
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
    return model

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # self.cnn = cnn
        model = get_vgg19(device='cuda')

        # make name consistent
        # print('start')
        self.block1_conv1 = model[0:2]  # [0]
        # print(self.block1_conv1)
        self.block1_conv2 = model[2:4]
        # print(self.block1_conv2)

        self.block2_conv1 = model[4:7]  # [2]
        # print(self.block2_conv1)
        self.block2_conv2 = model[7:9]
        # print(self.block2_conv2)

        self.block3_conv1 = model[9:12]  # [4]
        # print(self.block3_conv1)
        self.block3_conv2 = model[12:14]
        # print(self.block3_conv2)
        self.block3_conv3 = model[14:16]
        # print(self.block3_conv3)
        self.block3_conv4 = model[16:18]
        # print(self.block3_conv4)

        self.block4_conv1 = model[18:21]  # [8]
        # print(self.block4_conv1)
        self.block4_conv2 = model[21:23]
        # print(self.block4_conv2)
        self.block4_conv3 = model[23:25]
        # print(self.block4_conv3)
        self.block4_conv4 = model[25:27]
        # print(self.block4_conv4)

        self.block5_conv1 = model[27:30]  # [12]
        # print(self.block5_conv1)
        self.block5_conv2 = model[30:32]
        # print(self.block5_conv2)

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # imagenet
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        inp = x.clone()
        # print('here:', inp.min(), inp.max())
        inp[:, 0:1, ...] = (x[:, 0:1, ...] - 0.485) / 0.229
        inp[:, 1:2, ...] = (x[:, 1:2, ...] - 0.456) / 0.224
        inp[:, 2:3, ...] = (x[:, 2:3, ...] - 0.406) / 0.225

        outputs = []
        x = self.block1_conv1(inp)
        outputs.append(x)
        # print(x.shape)
        x = self.block1_conv2(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block2_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block2_conv2(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block3_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv2(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv3(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv4(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block4_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv2(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv3(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv4(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block5_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block5_conv2(x)
        outputs.append(x)
        # print(x.shape)

        return outputs

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # 如果x，y的通道数不是3，那么就扩展成3
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if y.shape[1] == 1:
            y = y.expand(-1, 3, -1, -1)

        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):

        # 如果x，y的通道数不是3，那么就扩展成3
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if y.shape[1] == 1:
            y = y.expand(-1, 3, -1, -1)

        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

class GANLoss(nn.Module):
    def __init__(self,  target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

    #这个我可以改，改成我自己定义的GAN
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var= self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if target_is_real:
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) +
                    torch.mean((y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD
        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) +
                    torch.mean((y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG




# contextual loss related
def isNone(x):
    return type(x) is type(None)


def feature_normalize(feature_in):
    """
    特征归一化
    :param feature_in:
    :return:
    """
    feature_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_norm)
    return feature_in_norm, feature_norm

def batch_patch_extraction(image_tensor, kernel_size, stride):
    """
    将输入的图像分割成patch
    [n, c, h, w] -> [n, np(num_patch), c, k, k]
    """
    n, c, h, w = image_tensor.shape
    h_out = math.floor((h - (kernel_size-1) - 1) / stride + 1)
    w_out = math.floor((w - (kernel_size-1) - 1) / stride + 1)
    unfold_tensor = F.unfold(image_tensor, kernel_size=kernel_size, stride=stride)
    unfold_tensor = unfold_tensor.contiguous().view(
        n, c * kernel_size * kernel_size, h_out, w_out)
    return unfold_tensor


def compute_cosine_distance(x, y):
    """
    计算余弦距离
    :param x:
    :param y:
    :return:
    """
    N, C, _, _ = x.size()

    # to normalized feature vectors
    # x_mean = x.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    y_mean = y.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    x, x_norm = feature_normalize(x - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    y, y_norm = feature_normalize(y - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    x = x.view(N, C, -1)
    y = y.view(N, C, -1)

    # cosine distance = 1 - similarity
    x_permute = x.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth

    # convert similarity to distance
    sim = torch.matmul(x_permute, y)
    dist = (1 - sim) / 2 # batch_size * feature_size^2 * feature_size^2

    return dist.clamp(min=0.)


def compute_l2_distance(x, y):
    """
    计算L2距离
    :param x:
    :param y:
    :return:
    """
    N, C, Hx, Wx = x.size()
    _, _, Hy, Wy = y.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s.unsqueeze(2).expand_as(A) - 2 * A + x_s.unsqueeze(1).expand_as(A)
    dist = dist.transpose(1, 2).reshape(N, Hx*Wx, Hy*Wy)
    dist = dist.clamp(min=0.) / C

    return dist


class GuidedCorrespondenceLoss(torch.nn.Module):
    '''
        纹理迁移损失
        input is Al, Bl, channel = 1, range ~ [0, 255]
        输入为（生成图，真实图，mask，方向图（用不上））
    '''

    def __init__(self,
                 sample_size=100, h=0.5, patch_size=7,
                 progression_weight=10.,
                 orientation_weight=0,
                 occurrence_weight=0):
        super(GuidedCorrespondenceLoss, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = max(patch_size // 2, 2)
        self.h = h

        self.vgg = VGG19().cuda()
        self.vgg.eval().requires_grad_(False)

        self.progression_weight = progression_weight
        self.orientation_weight = orientation_weight
        self.occurrence_weight = occurrence_weight

    def feature_extraction(self, feature, sample_field=None):
        # Patch extraction - use patch as single feature
        if self.patch_size > 1:
            feature = batch_patch_extraction(feature, self.patch_size, self.stride)

        # Random sampling - random patches
        num_batch, num_channel = feature.shape[:2]
        if num_batch * feature.shape[-2] * feature.shape[-1] > self.sample_size ** 2:
            if isNone(sample_field):
                sample_field = torch.rand(
                    num_batch, self.sample_size, self.sample_size, 2, device=feature.device) * 2 - 1
            feature = F.grid_sample(feature, sample_field, mode='nearest', align_corners=True)

        # Concatenate tensor
        sampled_feature = feature

        return sampled_feature, sample_field

    def calculate_distance(self, target_features, refer_features, progressions=None, orientations=None):
        origin_target_size = target_features.shape[-2:]
        origin_refer_size = refer_features.shape[-2:]

        # feature
        target_features, target_field = self.feature_extraction(target_features)
        refer_features, refer_field = self.feature_extraction(refer_features)
        d_total = compute_cosine_distance(target_features, refer_features)

        # progression
        use_progression = self.progression_weight > 0 and not isNone(progressions)
        if use_progression:
            with torch.no_grad():
                target_prog, refer_prog = progressions  # resize progression to corresponding size
                target_prog = F.interpolate(target_prog, origin_target_size)
                refer_prog = F.interpolate(refer_prog, origin_refer_size)

                target_prog = self.feature_extraction(target_prog, target_field)[0]
                refer_prog = self.feature_extraction(refer_prog, refer_field)[0]

                d_prog = compute_l2_distance(target_prog, refer_prog)
            d_total += d_prog * self.progression_weight

        # orientation
        use_orientation = self.orientation_weight > 0 and not isNone(orientations)
        if use_orientation:
            with torch.no_grad():
                target_orient, refer_orient = orientations
                target_orient = F.interpolate(target_orient, origin_target_size)
                refer_orient = F.interpolate(refer_orient, origin_refer_size)

                target_orient = self.feature_extraction(target_orient, target_field)[0]
                refer_orient = self.feature_extraction(refer_orient, refer_field)[0]
                target_orient = target_orient.view(target_orient.shape[0], 2, self.patch_size ** 2,
                                                   target_orient.shape[-2], target_orient.shape[-1])
                refer_orient = refer_orient.view(refer_orient.shape[0], 2, self.patch_size ** 2,
                                                 refer_orient.shape[-2], refer_orient.shape[-1])

                d_orient = 0
                for i in range(self.patch_size ** 2):
                    d_orient += torch.min(
                        compute_l2_distance(target_orient[:, :, i], refer_orient[:, :, i]),
                        compute_l2_distance(target_orient[:, :, i], -refer_orient[:, :, i])
                    )
                d_orient /= self.patch_size ** 2
            d_total += d_orient * self.orientation_weight

        min_idx_for_target = torch.min(d_total, dim=-1, keepdim=True)[1]

        # occurrence penalty
        use_occurrence = self.occurrence_weight > 0
        if use_occurrence:
            with torch.no_grad():
                omega = d_total.shape[1] / d_total.shape[2]
                occur = torch.zeros_like(d_total[:, 0, :])
                indexs, counts = min_idx_for_target[0, :, 0].unique(return_counts=True)
                occur[:, indexs] = counts / omega
                occur = occur.view(1, 1, -1)
                d_total += occur * self.occurrence_weight

        return d_total

    def calculate_loss(self, d):
        # --------------------------------------------------
        # minimize closest distance
        # --------------------------------------------------
        # loss = d.min(dim=-1)[0].mean()

        # --------------------------------------------------
        # guided correspondence loss
        # --------------------------------------------------
        # calculate loss
        # for each target feature, find closest refer feature
        d_min = torch.min(d, dim=-1, keepdim=True)[0]
        # convert to relative distance
        d_norm = d / (d_min + sys.float_info.epsilon)
        w = torch.exp((1 - d_norm) / self.h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # texture loss per sample
        CX = torch.max(A_ij, dim=-1)[0]
        loss = -torch.log(CX).mean()

        # --------------------------------------------------
        # contextual loss
        # --------------------------------------------------
        # # calculate contextual similarity and contextual loss
        # d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + sys.float_info.epsilon)  # batch_size * feature_size^2 * feature_size^2
        # # pairwise affinity
        # w = torch.exp((1 - d_norm) / self.h)
        # A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # # contextual loss per sample
        # CX = torch.max(A_ij, dim=1)[0].mean(dim=-1)
        # loss = -torch.log(CX).mean()

        return loss

    def forward(self, target_features, refer_features, mask=None, orientations=None):
        """

        :param gen_img: 生成的图像
        :param real_img: 真实的图像
        :param mask: mask图像
        :param orientations: 方向图像（用不上）
        :return:
        """

        progressions = [mask, mask] if mask is not None else None

        d_total = self.calculate_distance(
            target_features, refer_features,
            progressions, orientations)
        loss = self.calculate_loss(d_total)

        return loss

    def get_loss(self, gen_img, real_img, mask=None, layer = [2,4,8]):

        total_loss = 0

        target_features = self.vgg(gen_img)
        refer_features = self.vgg(real_img)
        progressions = [mask, mask] if mask is not None else None

        for i in layer:
            d_total = self.calculate_distance(
                target_features[i], refer_features[i],
                progressions)
            total_loss += self.calculate_loss(d_total)

        return total_loss

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        cnum = 64
        self.encoder = nn.Sequential(
            use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum * 2,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum * 2, out_channels=cnum * 4,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum * 4, out_channels=cnum * 8,
                                        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Conv2d(in_channels=cnum * 8, out_channels=1, kernel_size=5, stride=1, padding=1)
        if init_weights:
            self.init_weights()


    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def forward(self, x):
        x = self.encoder(x)
        label_x = self.classifier(x)
        if self.use_sigmoid:
            label_x = torch.sigmoid(label_x)
        return label_x


def ssim(x, y, window_size=11, window_sigma=1.5, data_range=1.0, size_average=True):
    # 创建高斯加权窗口
    channel = x.size(1)
    window = torch.FloatTensor(channel, 1, window_size, window_size).to(x.device)
    window = window.repeat(1, channel, 1, 1)
    window.data.normal_(0, window_sigma)

    # 计算亮度相似性、对比度相似性和结构相似性
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channel)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu_x_mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2)) / (
                (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ms_ssim(x, y, window_size=11, window_sigma=1.5, data_range=1.0, size_average=True, level=5):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(x.device)
    msssim = torch.zeros(level).to(x.device)
    mcs = torch.zeros(level).to(x.device)

    for i in range(level):
        ssim_map = ssim(x, y, window_size, window_sigma, data_range, size_average=False)
        msssim[i] = torch.clamp(ssim_map.mean(), 0, 1)
        mcs[i] = torch.clamp(ssim_map.std(), 0, 1)

        if i < level - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    mcs[-1] = 1

    ms_ssim_val = torch.prod((msssim ** weights) * (mcs ** (1 - weights)))

    return ms_ssim_val


# 计算mask边缘损失
class MaskEdgeLoss(nn.Module):
    def __init__(self, input_channels = 1, scale = 7, mask_num = 0, data_range = 2, use_sigmoid=False):
        super(MaskEdgeLoss, self).__init__()
        self.scale = scale
        self.mask_num = mask_num
        self.use_sigmoid = use_sigmoid

        self.loss = MS_SSIM(data_range=data_range, channel=input_channels, size_average=True)

    def forward(self, real_img, gen_img, mask):


        # 对mask进行膨胀
        mask_dilate = dilate_mask(mask, self.scale, self.mask_num)

        # 对mask进行腐蚀
        mask_erode = shrunk_mask(mask, self.scale, self.mask_num)

        # 获取mask的边缘
        if self.mask_num == 1:
            mask_edge = mask_dilate - mask_erode
        else:
            mask_edge = mask_erode - mask_dilate
            mask_edge = 1 - mask_edge

        # 是否要对图像进行sigmoid
        if self.use_sigmoid:
            real_img = torch.sigmoid(real_img)
            gen_img = torch.sigmoid(gen_img)


        # 获取mask_edge部分
        if self.mask_num == 1:
            real_img = real_img * mask_edge
            gen_img = gen_img * mask_edge
        else:
            real_img = real_img * (1 - mask_edge)
            gen_img = gen_img * (1 - mask_edge)


        loss = self.loss(real_img, gen_img)

        return 1 - loss


class StructuralLoss(nn.Module):
    def __init__(self, scale = 11, mask_num = 0):
        super(StructuralLoss, self).__init__()
        self.scale = scale
        self.mask_num = mask_num


    def forward(self, real_edge, gen_edge, mask):

        # TODO: 结构偏移损失
        loss = 0
        return loss


class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.loss = lpips.LPIPS(net='vgg').cuda()

    def forward(self, x, y):
        return self.loss(x, y).mean()


class BinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(BinarizationLoss, self).__init__()

    def forward(self, input, mask = None):

        # 计算二值化损失
        loss = torch.norm(torch.abs(torch.abs(input) - 1), p=2)

        # 如果mask不为空，那么只计算mask部分的均值
        if mask is not None:
            loss = loss * (1-mask)
            loss = loss.sum() / ((1-mask).sum() + sys.float_info.epsilon)
        else:
            loss = loss.mean()

        return loss.mean()






if __name__ == '__main__':
    img_path = r'E:\CODE\project\manga\comic-inpaint\test_image\image.jpg'
    line_path = r'E:\CODE\project\manga\comic-inpaint\test_image\structure.png'
    mask_path = r'E:\CODE\project\manga\comic-inpaint\test_image\mask.jpg'

    # 读取图片（灰度）
    img = Image.open(img_path).convert('L')
    line = Image.open(line_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    # resize
    img = img.resize((256, 256))
    line = line.resize((256, 256))
    mask = mask.resize((256, 256))

    # 变为tensor
    img = transforms.ToTensor()(img)
    line = transforms.ToTensor()(line)
    mask = transforms.ToTensor()(mask)

    # mask二值化
    mask = (mask > 0.5).float()
    # mask = 1-mask

    # img在mask部分填充为1
    img_mask = torch.where(mask > 0, 1, img)


    img = img.unsqueeze(0)
    line = line.unsqueeze(0)
    img_mask = img_mask.unsqueeze(0)
    mask = mask.unsqueeze(0)

    plt.imshow(img_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.show()

    loss = MaskEdgeLoss(mask_num=1)
    loss_2 = StructuralLoss(mask_num=1)
    loss_3 = LPIPSLoss()



    x = loss(img, img_mask, mask)
    y = loss_2(line, line, mask)
    z = loss_3(img, img_mask)

    print(x)
    print(y)
    print(z)

