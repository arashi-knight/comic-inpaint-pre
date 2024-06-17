import torch
from torchvision.utils import save_image

import myUtils
from config import Config
from edge_detector.model_torch import res_skip
from model.ComicUnet_v2 import ComicNet_v2
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from myUtils import NormalizeMask, ReverseMask


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'image')
        self.structure_dir = os.path.join(root_dir, 'structure')
        self.mask_dir = os.path.join(root_dir, 'mask')

        self.image_files = os.listdir(self.image_dir)
        self.structure_files = os.listdir(self.structure_dir)
        self.mask_files = os.listdir(self.mask_dir)

        # 假设图片、结构和掩模文件名字是一致的
        self.image_files.sort()
        self.structure_files.sort()
        self.mask_files.sort()
        image_size = (512, 512)
        # 定义转换
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # 将图像转换为张量
            # 添加其他转换如归一化等
            transforms.Normalize((0.5), (0.5))
        ])

        self.structure_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # 将图像转换为张量
            # 添加其他转换如归一化等
            transforms.Normalize((0.5), (0.5))
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # 将图像转换为张量
            # 添加其他转换如归一化等
            NormalizeMask(threshold=0.2),
            # ReverseMask()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        structure_name = os.path.join(self.structure_dir, self.structure_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

        # 使用PIL读取图像
        image = Image.open(img_name).convert('L')
        structure = Image.open(structure_name).convert('L')
        mask = Image.open(mask_name).convert('L')

        # 应用转换
        image = self.image_transform(image)
        structure = self.structure_transform(structure)
        mask = self.mask_transform(mask)

        return image, structure, mask

def get_edge(edge_model, img):
    """
    获取边缘
    :param img: 图片
    :return: 边缘
    """
    # with torch.no_grad():

    # 将-1到1的图片放缩到0-255
    img_x = (img + 1) * 127.5

    edge = edge_model(img_x)

    # 截取255-0
    edge = torch.clamp(edge, 0, 255)

    # 放缩到-1至1
    edge = (edge - 127.5) / 127.5

    return edge


def get_grid(imgs, structures, masks, comp_imgs, comp_imgs_structures):
    # print('imgs:', imgs)
    # print('comp_imgs:', comp_imgs)

    # 都转成rgb格式
    imgs_rgb = myUtils.gray2rgb(imgs)
    structures_rgb = myUtils.gray2rgb(structures)
    masks_rgb = myUtils.gray2rgb(masks)
    comp_imgs_rgb = myUtils.gray2rgb(comp_imgs)
    comp_imgs_structures_rgb = myUtils.gray2rgb(comp_imgs_structures, mode='RED')
    mask_red = myUtils.gray2rgb(masks, mode='RED')
    # 从【0,1】放缩到【-1,1】
    mask_red = (mask_red - 0.5) / 0.5
    # masks_rgb = (masks_rgb - 0.5) / 0.5
    # print('comp_imgs__rgb:', comp_imgs_rgb)
    # print('imgs_rgb', imgs_rgb)
    # 在img的mask区域填充为红色
    img_masked_red = torch.where(masks.byte() == False, mask_red, imgs)  # 将 mask 区域的像素值设为红色 (1, 0, 0)

    # 拼接structures和comp_imgs_structures的mask区域
    comp_imgs_structures_rgb_x = comp_imgs_structures_rgb * (1 - masks_rgb) + structures_rgb * masks_rgb

    # 转换到0-1
    imgs_rgb = (imgs_rgb + 1) / 2
    structures_rgb = (structures_rgb + 1) / 2
    masks_rgb = (masks_rgb + 1) / 2
    # img_masked_rgb = (img_masked_rgb + 1) / 2
    img_masked_red = (img_masked_red + 1) / 2
    # print('comp_imgs__rgb:', comp_imgs_rgb)
    comp_imgs_rgb = (comp_imgs_rgb + 1) / 2
    # print('comp_imgs__rgb:', comp_imgs_rgb)
    # print('imgs_rgb', imgs_rgb)
    # comp_imgs_structures_rgb = (comp_imgs_structures_rgb + 1) / 2
    comp_imgs_structures_rgb_x = (comp_imgs_structures_rgb_x + 1) / 2

    grid_list = [imgs_rgb, structures_rgb, masks_rgb, img_masked_red, comp_imgs_rgb, comp_imgs_structures_rgb_x]

    return myUtils.make_val_grid_list(grid_list)

def test_model(model, dataloader, edge_model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    outpu_comp_path = os.path.join(output_path, 'comp')
    if not os.path.exists(outpu_comp_path):
        os.makedirs(outpu_comp_path)

    model.eval()
    edge_model.eval()

    for i, data in enumerate(dataloader):
        image, structure, mask = data
        # cuda
        image = image.cuda()
        structure = structure.cuda()
        mask = mask.cuda()

        image_mask = image * mask
        structure_mask = structure * mask


        with torch.no_grad():
            gen_imgs = model(image_mask, structure_mask, mask)

            comp_imgs = (image * mask) + (gen_imgs * (1 - mask))

            # 获取边缘
            edge = get_edge(edge_model, comp_imgs)

            # 构造输出图(原图，结构图，mask，mask遮罩，输出图，输出结构图)
            grid = get_grid(image, structure, mask, comp_imgs, edge)

            # print('comp_imgs:', comp_imgs.shape)
            # print_max_min(comp_imgs)
            # print('mask:', mask.shape)
            # print_max_min(mask)
            # print_unique(mask)

            # 保存输出图
            save_image(grid, os.path.join(output_path, f'{i}.png'))
            #comp从[-1,1]缩放到0-1
            comp_imgs = (comp_imgs + 1) / 2
            save_image(comp_imgs, os.path.join(outpu_comp_path, f'{i}_comp.png'))

def get_edge_model(config):
    """
    获取边缘检测
    :return: 模型
    """
    # 获取边缘检测
    edge_detect = res_skip()

    edge_detect.load_state_dict(torch.load(config.edge_model_path))

    myUtils.set_requires_grad(edge_detect, False)

    edge_detect.cuda()
    edge_detect.eval()

    return edge_detect

#对input路径下的所有图片，输入edge_model，并将输出保存到output文件夹
def get_edge_by_path(input, output, edge_model):
    if not os.path.exists(output):
        os.makedirs(output)

    # 遍历input文件夹下的所有图片
    for file in os.listdir(input):
        if file.endswith('.jpg'):
            # 读取图像
            img = Image.open(os.path.join(input, file)).convert('L')
            img = transforms.ToTensor()(img).cuda().unsqueeze(0)
            print('img:', img.shape)
            #缩放到0-255
            img = img * 255

            # 获取边缘
            edge = edge_model(img).clamp(0, 255)

            #缩放到0,1
            edge = edge / 255.0
            # edge = (edge - 127.5) / 127.5


            # 保存边缘
            save_image(edge, os.path.join(output, file))


# 输出张量的最大值和最小值
def print_max_min(tensor):
    print('max:', tensor.max().item(), 'min:', tensor.min().item())

# 输出张量中每个独特值
def print_unique(tensor):
    print('unique:', np.unique(tensor.cpu().numpy()))


# 输出两张图片a，b和mask，分别将ab转换为结构图，将a的mask区域转换为红色并与b的非mask区域拼接
def concat_a_b_structure(a, b, mask, edge_model):
    # 把b，mask放缩到a的大小
    b = transforms.Resize(a.shape[-1])(b)
    mask = transforms.Resize(a.shape[-1])(mask)

    # a转换为红色，b转换为灰色
    a_red = myUtils.gray2rgb(a, mode='RED')
    b_gray = myUtils.gray2rgb(b)
    mask = myUtils.gray2rgb(mask)

    # 拼接a和b的mask区域
    a_masked_b = a_red * (1 - mask) + b * mask

    return a_masked_b

def concat_a_b_structure_by_path(a_path, b_path, mask_path, edge_model):
    a = Image.open(a_path).convert('L')
    b = Image.open(b_path).convert('L')
    mask = Image.open(mask_path).convert('L')


    a = transforms.ToTensor()(a).cuda().unsqueeze(0)
    b = transforms.ToTensor()(b).cuda().unsqueeze(0)
    mask = transforms.ToTensor()(mask).cuda().unsqueeze(0)

    # 将a,b转换为结构图
    a = a*255.
    b = b*255.

    a = edge_model(a)
    b = edge_model(b)

    a = torch.clamp(a, 0, 255)/255.
    b = torch.clamp(b, 0, 255)/255.

    a_masked_b = concat_a_b_structure(a, b, mask, edge_model)

    return a_masked_b

def concat_a_b_structure_by_path_v2(a_path, b_path, mask_path, edge_model):
    a = Image.open(a_path).convert('L')
    b = Image.open(b_path).convert('L')
    mask = Image.open(mask_path).convert('L')


    a = transforms.ToTensor()(a).cuda().unsqueeze(0)
    b = transforms.ToTensor()(b).cuda().unsqueeze(0)
    mask = transforms.ToTensor()(mask).cuda().unsqueeze(0)

    # 将a,b转换为结构图
    a = a*255.

    a = edge_model(a)

    a = torch.clamp(a, 0, 255)/255.

    a_masked_b = concat_a_b_structure(a, b, mask, edge_model)

    return a_masked_b


if __name__ == '__main__':
    config = Config()
    model = ComicNet_v2(in_channels=1, out_channels=1).cuda()
    edge_model = get_edge_model(config)

    model.load_state_dict(torch.load(r'C:\Users\73631\Downloads\g_model_120.pth'), strict=False)

    dataset = CustomDataset(r'E:\CODE\dataset\Comic\test_comic')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input = r'E:\CODE\dataset\Comic\test_comic\image'
    output = r'D:\BaiduSyncdisk\漫画修复\数据集相关\关键帧文件\修复输出\ours'

    test_model(model, dataloader, edge_model, output)








