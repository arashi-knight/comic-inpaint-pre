import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist

from skimage.transform import resize
from torchvision.transforms import transforms


def distance_transform(image_tensor, scale = 13, bias = 0.):
    # 将输入图像转换为二进制图像（0和1）
    # binary_image = torch.where(image_tensor > 0.5, torch.ones_like(image_tensor), torch.zeros_like(image_tensor))
    # binary_image = image_tensor
    binary_image = torch.sigmoid(image_tensor)
    # 计算距离场
    distance_transform = F.conv2d(binary_image, torch.ones(1, 1, scale, scale), padding=1)
    distance_transform = torch.sqrt(distance_transform)

    # 归一化到[0, 1]
    distance_transform = distance_transform - torch.min(distance_transform)
    distance_transform = distance_transform / torch.max(distance_transform)

    # distance_transform不为0的地方加上一个bias
    distance_transform = distance_transform + bias

    distance_transform = torch.clamp(distance_transform, 0, 1)

    return distance_transform



if __name__ == '__main__':

    line_path = r'E:\CODE\project\manga\comic-inpaint\test_image\structure.png'

    x = Image.open(line_path).convert('L')

    # resize
    x = x.resize((256, 256))

    # 张量化
    x = transforms.ToTensor()(x)

    # 放缩到-1, 1
    x = 2 * x - 1


    md = distance_transform(x)
    uni_list = np.unique(md.squeeze().cpu().numpy())
    print(uni_list)
    # md = 1-md

    # 作为图像show一下md
    md = md.squeeze().cpu().numpy()
    plt.imshow(md, cmap='gray')
    plt.show()