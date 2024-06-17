
import os
import time
from time import sleep

import torch
from colorama import Fore
from torch import nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import torch.nn.functional as F
import myUtils
from config import Config
from data import comic_dataloader
from data.dataloader_init import get_dataloader
from edge_detector.model_torch import res_skip
from model.ComicUnet import ComicNet
from model.ComicUnet_v2 import ComicNet_v2
from model.ComicUnet_v3 import ComicNet_v3
from model.ComicUnet_v4 import ComicNet_v4
from model.ComicUnet_v5 import ComicNet_v5
from model.cilp import FrozenCLIPEmbedder
from model.discriminator.t_s_discriminator import T_S_Discriminator
from model.loss import GANLoss, PerceptualLoss, StyleLoss, GuidedCorrespondenceLoss, Discriminator, MaskEdgeLoss, \
    StructuralLoss, LPIPSLoss, BinarizationLoss

from myUtils import set_device, get_optimizer, get_optimizer_D, ssim_by_list, psnr_by_list, make_val_grid

class Trainer_Our():
    def __init__(self, config: Config, is_test = False, debug = False):
        self.config = config
        # self.device = 'cuda' if config.is_cuda else 'cpu'
        self.epoch = 0
        # 迭代次数
        self.iteration = 0

        self.debug = debug

        self.val_img_save_path_compare = config.val_img_save_path_compare
        self.val_img_save_path_single = config.val_img_save_path_single
        self.best_val_img_save_path_compare = config.best_val_img_save_path_compare
        self.best_val_img_save_path_single = config.best_val_img_save_path_single
        self.val_from_train_img_save_path_compare = config.val_from_train_img_save_path_compare
        self.log_path = config.log_path
        self.debug_log_path = config.debug_log_path
        self.model_path = config.model_path

        self.best_psnr = 0
        self.best_ssim = 0
        self.data_range = config.data_range

        print('正在初始化数据集')
        self.classes, self.train_dataloader, self.test_dataloader, self.val_dataloader, self.val_from_train_dataloader = get_dataloader(config)
        print('数据集初始化完成')
        # self.classes_num = len(self.classes)

        # 损失函数

        self.adversarial_loss = GANLoss(tensor=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor)
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.lpips_loss = LPIPSLoss()

        self.texture_loss = GuidedCorrespondenceLoss()
        self.mask_edge_loss = MaskEdgeLoss(input_channels=1,scale=11,mask_num=0,data_range=config.data_range)
        self.structural_loss = StructuralLoss(scale=11, mask_num=0)
        self.bin_loss = BinarizationLoss()

        self.avg_loss_hole = 0
        self.avg_loss_valid = 0
        self.avg_loss_l1 = 0
        self.avg_loss_adversarial = 0
        self.avg_loss_perceptual = 0
        self.avg_loss_style = 0
        # 创新点损失
        self.avg_loss_texture = 0
        self.avg_loss_mask_edge = 0
        self.avg_loss_structural = 0


        print('正在初始化生成模型')
        # 模型
        # self.g_model = set_device(ComicNet(in_channels=config.input_channels, out_channels=config.output_channels,
        #                                    context_channels=config.context_channels, base_block_num=config.base_block_num))
        # 去掉融合
        self.g_model = set_device(ComicNet_v2(in_channels=config.input_channels, out_channels=config.output_channels,
                                              context_channels=config.context_channels, base_block_num=config.base_block_num))
        # 去掉纹理
        # self.g_model = set_device(ComicNet_v4(in_channels=config.input_channels, out_channels=config.output_channels,
        #                                       context_channels=config.context_channels, base_block_num=config.base_block_num))

        # 初始化边缘提取模型
        self.edge_model = set_device(self.get_edge_model())

        print('生成模型初始化完成')
        print('正在初始化clip模型')
        # clip模型
        # self.clip_model = set_device(FrozenCLIPEmbedder(version=config.clip_version, layer=config.clip_layer, layer_idx=config.clip_skipping))
        # self.clip_model = None
        print('clip模型初始化完成')




        if self.config.is_gray:
            self.d_model = set_device(
                T_S_Discriminator(image_in_channels=1,edge_in_channels=1))
        else:
            self.d_model = set_device(
                T_S_Discriminator(image_in_channels=1,edge_in_channels=1))

        # 优化器
        self.optimizer_g = get_optimizer(config, self.g_model)
        self.optimizer_d = get_optimizer_D(config, self.d_model)
        self.lr_decay_epoch = config.lr_decay_epoch
        self.step_lr_decay_factor = config.step_lr_decay_factor

        # 设置学习率衰减
        # self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=config.step_size, gamma=config.gamma)

        # 如果保存路径不存在则创建
        if not os.path.exists(self.val_img_save_path_compare):
            os.makedirs(self.val_img_save_path_compare)
        if not os.path.exists(self.val_img_save_path_single):
            os.makedirs(self.val_img_save_path_single)
        if not os.path.exists(self.best_val_img_save_path_compare):
            os.makedirs(self.best_val_img_save_path_compare)
        if not os.path.exists(self.best_val_img_save_path_single):
            os.makedirs(self.best_val_img_save_path_single)
        if not os.path.exists(self.val_from_train_img_save_path_compare):
            os.makedirs(self.val_from_train_img_save_path_compare)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


        self.val_test_img_save_path = os.path.join(config.result_path, 'val_test', 'val')
        self.val_test_from_train_img_save_path = os.path.join(config.result_path, 'val_test', 'val_from_train')
        if not os.path.exists(self.val_test_img_save_path):
            os.makedirs(self.val_test_img_save_path)
        if not os.path.exists(self.val_test_from_train_img_save_path):
            os.makedirs(self.val_test_from_train_img_save_path)

        # 加载模型
        self.load_model()

    def save_model_log(self):
        """
        保存模型各项参数数据
        :return:
        """
        # 所有参数量为
        g_params = myUtils.count_parameters(self.g_model)
        # 可训练参数量为
        g_params_trainable = myUtils.count_trainable_parameters(self.g_model)

        # 写入log
        myUtils.write_log('g_model参数量为：{}'.format(g_params/1e6), self.log_path)
        myUtils.write_log('g_model可训练参数量为：{}'.format(g_params_trainable/1e6), self.log_path)

        # 鉴别器
        d_params = myUtils.count_parameters(self.d_model)
        d_params_trainable = myUtils.count_trainable_parameters(self.d_model)

        # 写入log
        myUtils.write_log('d_model参数量为：{}'.format(d_params/1e6), self.log_path)
        myUtils.write_log('d_model可训练参数量为：{}'.format(d_params_trainable/1e6), self.log_path)


    # 获取学习率
    def get_lr(self, type='G'):
        if type == 'G':
            return self.optimizer_g.param_groups[0]['lr']
        return self.optimizer_d.param_groups[0]['lr']

    # 调整学习率（感觉没啥用）
    def adjust_learning_rate(self, scale):
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] *= scale
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] *= scale

    # 保存模型
    def save_model(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model.pth')
        d_model_path = os.path.join(self.model_path, 'd_model.pth')
        torch.save(self.g_model.state_dict(), g_model_path)
        torch.save(self.d_model.state_dict(), d_model_path)

    # 保存第x个epoch的模型
    def save_model_epoch(self, epoch):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_{}.pth'.format(epoch))
        d_model_path = os.path.join(self.model_path, 'd_model_{}.pth'.format(epoch))
        torch.save(self.g_model.state_dict(), g_model_path)
        torch.save(self.d_model.state_dict(), d_model_path)

    def save_model_last(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_last.pth')
        d_model_path = os.path.join(self.model_path, 'd_model_last.pth')
        torch.save(self.g_model.state_dict(), g_model_path)
        torch.save(self.d_model.state_dict(), d_model_path)

    def load_model_epoch(self, epoch):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_{}.pth'.format(epoch))
        d_model_path = os.path.join(self.model_path, 'd_model_{}.pth'.format(epoch))
        self.g_model.load_state_dict(torch.load(g_model_path))
        self.d_model.load_state_dict(torch.load(d_model_path))

    def load_model_last(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_last.pth')
        d_model_path = os.path.join(self.model_path, 'd_model_last.pth')
        self.g_model.load_state_dict(torch.load(g_model_path))
        self.d_model.load_state_dict(torch.load(d_model_path))

    # 一轮训练
    def train_epoch(self):

        # 平均l1损失
        self.avg_loss_l1 = 0
        batch_count = 0

        # 清空debug_log
        myUtils.clear_log(self.debug_log_path)
        # 当前epoch
        myUtils.write_log('epoch:{}'.format(self.epoch), self.debug_log_path, print_log=False)
        # 每个epoch的进度条
        with tqdm(total=len(self.train_dataloader),
                  bar_format=Fore.BLACK + '|{bar:30}|正在进行训练|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as train_pbar:
            for i, (imgs, structures, masks, labels, tags) in enumerate(self.train_dataloader):
                if self.debug:
                    myUtils.write_log('batch:{}'.format(i), self.debug_log_path, bar=train_pbar, print_log=False)
                # 设置cuda
                imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                # 拼接图片和mask
                img_masked = imgs * masks
                structures_masked = structures * masks

                context_encode = self.get_context_encode(tags, labels)

                # 生成图片
                gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                if self.debug:
                    myUtils.write_log('图片生成完成，gen_imgs:{}'.format(gen_imgs.shape), self.debug_log_path, bar=train_pbar, print_log=False)
                # 补全图片(把mask部分的生成图片和原图拼接)
                comp_imgs = imgs * masks + gen_imgs * (1 - masks)


                # 优化参数
                self.optimize_parameters(imgs, structures, gen_imgs, masks)

                self.avg_loss_l1 += self.loss_l1.item()
                if self.debug:
                    myUtils.write_log('优化参数完成', self.debug_log_path, bar=train_pbar, print_log=False)

                batch_count += 1
                train_pbar.update(1)

            self.avg_loss_l1 = self.avg_loss_l1 / batch_count

    # 训练
    def train(self):
        with tqdm(total=self.config.epochs,
                  bar_format=Fore.MAGENTA + '|{bar:30}|当前epoch:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|{desc}') as epoch_pbar:

            psnr_list = []
            ssim_list = []
            lpips_list = []
            train_l1_list = []
            test_l1_list = []


            # 设置训练模式
            self.g_model.train()
            self.d_model.train()
            start_time = time.time()
            for epoch in range(self.epoch, self.config.epochs):
                epoch_pbar.update(1)
                # 进行一轮训练
                self.train_epoch()

                # 记录l1损失
                train_l1_list.append(self.avg_loss_l1)
                # 保存train_l1_loss折线图
                myUtils.draw_by_list(train_l1_list, 'TRAIN_L1', save_path=self.config.train_l1_loss_img_save_path, show_min=True)
                # 输出学习率
                myUtils.write_log('当前g_model学习率：{}'.format(self.get_lr()), self.log_path, bar=epoch_pbar)
                # 输出损失
                myUtils.write_log('epoch:{}, loss_l1:{}, loss_g:{}, loss_adversarial:{}, loss_perceptual:{}, loss_style:{}, loss_texture:{}'.format
                                  (epoch, self.avg_loss_l1, self.loss_g, self.loss_adversarial,
                                   self.loss_perceptual, self.loss_style, self.loss_texture), self.log_path, bar=epoch_pbar)

                # 记录创新点损失
                myUtils.write_log('epoch:{}, loss_mask_edge:{}, loss_structural:{}'.format(epoch, self.loss_mask_edge, self.loss_structural), self.log_path, bar=epoch_pbar)
                # 是否更新学习率，每隔一定epoch更新一次
                if (epoch+1) % self.lr_decay_epoch == 0:
                    self.adjust_learning_rate(self.step_lr_decay_factor)

                epoch_pbar.display()
                # 是否进行测试
                if (epoch+1) % self.config.test_interval == 0:
                    psnr, ssim, l1, lpips = self.test()
                    # 保存psnr和ssim
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    lpips_list.append(lpips)
                    test_l1_list.append(l1)
                    # 打印psnr和ssim
                    myUtils.draw_by_list(psnr_list, 'PSNR', save_path=self.config.psnr_img_save_path, show_max=True)
                    myUtils.draw_by_list(ssim_list, 'SSIM', save_path=self.config.ssim_img_save_path, show_max=True)
                    myUtils.draw_by_list(lpips_list, 'LPIPS', save_path=self.config.lpips_img_save_path, show_min=True)
                    myUtils.draw_by_list(test_l1_list, 'TEST_L1', save_path=self.config.test_l1_loss_img_save_path, show_min=True)

                    # 判断psnr和ssim是否大于之前的最优值
                    if psnr > self.best_psnr and ssim > self.best_ssim:
                        myUtils.write_log('获得最优，当前epoch:{},psnr:{},ssim:{},l1:{},lpips:{}'.format(epoch, psnr, ssim, l1, lpips), self.log_path,
                                        bar=epoch_pbar)
                        self.best_psnr = psnr
                        self.best_ssim = ssim
                        # 保存模型
                        self.save_model()
                        # 保存验证图片
                        myUtils.write_log('正在保存验证图片', self.log_path, bar=epoch_pbar)
                        self.val(best=True)
                    else:
                        myUtils.write_log('未提高，当前epoch:{},psnr:{},ssim:{}，l1:{},lpips:{}'.format(epoch, psnr, ssim, l1, lpips), self.log_path,
                                        bar=epoch_pbar)
                        # 保存验证图片
                        myUtils.write_log('正在保存验证图片', self.log_path, bar=epoch_pbar)
                        self.val()



                # 保存模型
                if (epoch+1) % self.config.save_interval == 0:
                    self.save_model_epoch(epoch)
                # 更新进度条
                # 获取时间数据
                _, time_left, time_end = myUtils.get_time(start_time, epoch, self.config.epochs)
                epoch_pbar.set_description('预计剩余时间:{}|预计结束时间:{}'.format(time_left, time_end))

            self.save_model_last()

    # 测试
    def test(self):

        with tqdm(total=len(self.test_dataloader),
                  bar_format=Fore.BLUE + '|{bar:30}|正在进行测试|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as test_pbar:
            avg_psnr = 0
            avg_ssim = 0
            avg_l1 = 0
            avg_lpips = 0
            batch_count = 0
            # 设置评估模式
            with torch.no_grad():
                self.g_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.test_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    # 拼接图片和mask
                    img_masked = imgs * masks
                    structures_masked = structures * masks

                    context_encode = self.get_context_encode(tags, labels)

                    # 生成图片
                    gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                    # 补全图片(把mask部分的生成图片和原图拼接)
                    comp_imgs = imgs * masks + gen_imgs * (1 - masks)


                    # 计算PSNR和SSIM
                    # 将图片转换为numpy
                    imgs_cpu = imgs.detach().cpu().numpy()
                    comp_imgs_cpu = comp_imgs.detach().cpu().numpy()

                    # 计算PSNR和SSIM
                    batch_psnr = psnr_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_ssim = ssim_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_l1 = F.l1_loss(imgs, gen_imgs).item()
                    batch_lpips = self.lpips_loss(imgs, comp_imgs).item()


                    # 计算平均PSNR和SSIM
                    avg_psnr += batch_psnr
                    avg_ssim += batch_ssim
                    avg_l1 += batch_l1
                    avg_lpips += batch_lpips
                    batch_count += 1
                    test_pbar.update(1)

        if batch_count == 0:
            batch_count = 1
        avg_psnr = avg_psnr / batch_count
        avg_ssim = avg_ssim / batch_count
        avg_l1 = avg_l1 / batch_count
        avg_lpips = avg_lpips / batch_count
        # print('avg_psnr:{}, avg_ssim:{}, avg_l1:{}'.format(avg_psnr, avg_ssim, avg_l1))
        return avg_psnr, avg_ssim, avg_l1, avg_lpips

    # 验证（保存验证图像）
    def val(self, best = False):
        # print('val')
        with tqdm(total=len(self.val_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.g_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    # 拼接图片和mask
                    img_masked = imgs * masks

                    structures_masked = structures * masks

                    context_encode = self.get_context_encode(tags, labels)

                    # 生成图片
                    gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                    # 补全图片(把mask部分的生成图片和原图拼接)
                    comp_imgs = (imgs * masks) + (gen_imgs * (1 - masks))

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, img_masked, comp_imgs)
                    if best:
                        compare_img_path = os.path.join(self.best_val_img_save_path_compare, f"{i:0>3d}" + '.jpg')
                        single_img_path = os.path.join(self.best_val_img_save_path_single, f"{i:0>3d}" + '.jpg')
                    else:
                        compare_img_path = os.path.join(self.val_img_save_path_compare, f"{i:0>3d}" + '.jpg')
                        single_img_path = os.path.join(self.val_img_save_path_single, f"{i:0>3d}" + '.jpg')

                    # 保存图片
                    save_image(val_grid, compare_img_path)
                    save_image(make_grid(comp_imgs, nrow=1, normalize=True, scale_each=True), single_img_path)

                    val_pbar.update(1)

        with tqdm(total=len(self.val_from_train_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行测试集验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.g_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_from_train_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    # 拼接图片和mask
                    img_masked = imgs * masks

                    structures_masked = structures * masks

                    context_encode = self.get_context_encode(tags, labels)

                    # 生成图片
                    gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                    # 补全图片(把mask部分的生成图片和原图拼接)
                    comp_imgs = (imgs * masks) + (gen_imgs * (1 - masks))

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, img_masked, comp_imgs)
                    compare_img_path = os.path.join(self.val_from_train_img_save_path_compare, f"{i:0>3d}" + '.jpg')
                    # 保存图片
                    save_image(val_grid, compare_img_path)

                    val_pbar.update(1)



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # 获取损失,输入（真实图片，生成图片，decode中间层，mask）
    def backward_d(self, real_img, real_edge, gen_img, gen_edge):
        """
        获取损失
        :param real_img: 真实图片
        :param gen_img: 生成图片
        :param mask: mask
        :return:
        """
        # print('get_loss')

        criterion = nn.BCELoss()

        # 鉴别器损失
        gen_img_d = self.d_model(gen_img, gen_edge)
        real_img_d = self.d_model(real_img, real_edge)

        # 创建真实目标值
        real_target = torch.full_like(real_img_d, 1.0).to(real_img_d.device)

        # 创建虚假目标值
        fake_target = torch.full_like(gen_img_d, 0.0).to(gen_img_d.device)


        self.loss_d = criterion(real_img_d, real_target) + criterion(gen_img_d, fake_target)

        self.loss_d = self.loss_d.mean()

        self.loss_d.backward()

    # 获取损失,输入（真实图片，生成图片，decode中间层，mask）
    def backward_g(self, real_img, real_edge, gen_img, gen_edge, mask):
        """
        获取损失
        :param real_img: 真实图片
        :param gen_img: 生成图片
        :param mask: mask
        :return:
        """

        criterion = nn.BCELoss()
        sigmod = nn.Sigmoid()
        # 生成器损失
        gen_img_d = self.d_model(gen_img, gen_edge)
        real_target = torch.full_like(gen_img_d, 1.0).to(gen_img_d.device)

        self.loss_adversarial = (criterion(gen_img_d, real_target) + criterion(sigmod(gen_edge), sigmod(real_edge))).mean()

        self.loss_g = self.loss_adversarial * self.config.loss_adversarial_weight

        # 生成器valid损失
        self.loss_valid = self.l1_loss(mask * gen_img, mask * real_img)
        self.loss_g += self.loss_valid / torch.mean(mask) * self.config.loss_valid_weight

        # 生成器hole损失
        self.loss_hole = self.l1_loss((1 - mask) * gen_img, (1 - mask) * real_img)
        self.loss_g += self.loss_hole / torch.mean(1 - mask) * self.config.loss_hole_weight

        # # 直接计算l1损失
        self.loss_l1 = self.loss_hole + self.loss_valid
        # self.loss_g += self.loss_l1 * self.config.loss_l1_weight

        # 生成器感知损失（perceptual_loss）
        self.loss_perceptual = self.perceptual_loss(gen_img, real_img)
        self.loss_g += self.loss_perceptual * self.config.loss_perceptual_weight

        # 生成器风格损失
        self.loss_style = self.style_loss(gen_img, real_img)
        self.loss_g += self.loss_style * self.config.loss_style_weight

        # 生成器纹理损失（创新点）
        self.loss_texture = self.texture_loss.get_loss((1 - mask) * gen_img, (1 - mask) * real_img)
        # self.loss_g += self.loss_texture * self.config.loss_texture_weight

        # 生成器结构损失（创新点）
        self.loss_mask_edge = self.mask_edge_loss(real_img, gen_img, mask)
        self.loss_g += self.loss_mask_edge * self.config.loss_mask_edge_weight

        # 边缘结构损失（创新点）
        self.loss_structural = self.structural_loss(real_edge, gen_edge, mask)
        self.loss_g += self.loss_structural * self.config.loss_structure_weight

        # 二值化损失
        self.loss_bin = self.bin_loss(gen_img, mask)
        self.loss_g += self.loss_bin * self.config.loss_bin_weight

        # 生成器总损失
        self.loss_g.backward()




    def optimize_parameters(self, real_img, real_edge, gen_img, mask):
        """
        优化参数
        :param real_img: 真实图片
        :param gen_img: 生成图片
        :param mask: mask
        :return:
        """

        # 计算gen_edge
        gen_edge = self.get_edge(gen_img)

        myUtils.set_requires_grad(self.d_model, True)
        # myUtils.set_requires_grad(self.g_model, False)
        # 更新鉴别器参数
        self.optimizer_d.zero_grad()
        self.backward_d(real_img, real_edge, gen_img.detach(), gen_edge.detach())
        self.optimizer_d.step()

        myUtils.set_requires_grad(self.d_model, False)
        # myUtils.set_requires_grad(self.g_model, True)
        # 更新生成器参数
        self.optimizer_g.zero_grad()
        self.backward_g(real_img, real_edge, gen_img, gen_edge, mask)
        self.optimizer_g.step()


    def load_model(self):
        print('load_model')

    def print_model_parm(self):
        print('print_model_parm')
        print('g_total:', sum(p.numel() for p in self.g_model.parameters()) / 1e6)
        print('d_total:', sum(p.numel() for p in self.d_model.parameters()) / 1e6)


    # 获取context_encode
    def get_context_encode(self, tags, labels):
        # # 获取context_encode
        # with torch.no_grad():
        #     context_encode = self.clip_model(tags)
        # # 拓展label维度
        # labels = labels.unsqueeze(1).expand(-1, 77, -1)
        # # 和label拼接
        # context_encode = torch.cat((context_encode, labels), dim=2)

        return None

    def get_edge_model(self):
        """
        获取边缘检测
        :return: 模型
        """
        # 获取边缘检测
        edge_detect = res_skip()

        edge_detect.load_state_dict(torch.load(self.config.edge_model_path))

        myUtils.set_requires_grad(edge_detect, False)

        edge_detect.cuda()
        edge_detect.eval()

        return edge_detect

    def get_edge(self, img):
        """
        获取边缘
        :param img: 图片
        :return: 边缘
        """
        # with torch.no_grad():

        # 将-1到1的图片放缩到0-255
        img = (img + 1) * 127.5

        edge = self.edge_model(img)

        # 截取255-0
        edge = torch.clamp(edge, 0, 255)

        # 放缩到-1至1
        edge = (edge - 127.5) / 127.5

        return edge

    def get_grid(self, imgs, structures, masks, img_masked, comp_imgs):

        comp_imgs_structures = self.get_edge(comp_imgs)

        # 都转成rgb格式
        imgs_rgb = myUtils.gray2rgb(imgs)
        structures_rgb = myUtils.gray2rgb(structures)
        masks_rgb = myUtils.gray2rgb(masks)
        img_masked_rgb = myUtils.gray2rgb(img_masked)
        comp_imgs_rgb = myUtils.gray2rgb(comp_imgs)
        comp_imgs_structures_rgb = myUtils.gray2rgb(comp_imgs_structures, mode='RED')
        mask_red = myUtils.gray2rgb(masks, mode='RED')
        # 从【0,1】放缩到【-1,1】
        mask_red = (mask_red - 0.5) / 0.5

        # 在img的mask区域填充为红色
        img_masked_red = torch.where(masks.byte() == False, mask_red, imgs)  # 将 mask 区域的像素值设为红色 (1, 0, 0)

        # 拼接structures和comp_imgs_structures的mask区域
        comp_imgs_structures_rgb_x = comp_imgs_structures_rgb * (1 - masks_rgb) + structures_rgb * masks_rgb

        # 从-1-1放缩到0-1
        # imgs_rgb = (imgs_rgb + 1) / 2
        # structures_rgb = (structures_rgb + 1) / 2
        # masks_rgb = masks_rgb
        # img_masked_red = (img_masked_red + 1) / 2
        # comp_imgs_rgb = (comp_imgs_rgb + 1) / 2
        # comp_imgs_structures_rgb_x = (comp_imgs_structures_rgb_x + 1) / 2

        grid_list = [imgs_rgb, structures_rgb, masks_rgb, img_masked_red, comp_imgs_rgb, comp_imgs_structures_rgb_x]



        return myUtils.make_val_grid_list(grid_list)

    def val_test(self, best = False):

        avg_psnr = 0
        avg_ssim = 0
        avg_l1 = 0
        avg_lpips = 0
        batch_count = 0
        # print('val')
        with tqdm(total=len(self.val_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.g_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    # 拼接图片和mask
                    img_masked = imgs * masks

                    structures_masked = structures * masks

                    context_encode = self.get_context_encode(tags, labels)

                    # 生成图片
                    gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                    # 补全图片(把mask部分的生成图片和原图拼接)
                    comp_imgs = (imgs * masks) + (gen_imgs * (1 - masks))

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, img_masked, comp_imgs)

                    compare_img_path = os.path.join(self.val_test_img_save_path, f"{i:0>3d}" + '.jpg')

                    # 保存图片
                    save_image(val_grid, compare_img_path)


                    # 計算psnr和ssim,lpips
                    # 将图片转换为numpy
                    imgs_cpu = imgs.detach().cpu().numpy()
                    comp_imgs_cpu = comp_imgs.detach().cpu().numpy()

                    # 计算PSNR和SSIM
                    batch_psnr = psnr_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_ssim = ssim_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_l1 = F.l1_loss(imgs, comp_imgs).item()
                    batch_lpips = self.lpips_loss(imgs, comp_imgs).item()


                    # 计算平均PSNR和SSIM
                    avg_psnr += batch_psnr
                    avg_ssim += batch_ssim
                    avg_l1 += batch_l1
                    avg_lpips += batch_lpips
                    batch_count += 1

                    val_pbar.update(1)
        avg_psnr = avg_psnr / batch_count
        avg_ssim = avg_ssim / batch_count
        avg_l1 = avg_l1 / batch_count
        avg_lpips = avg_lpips / batch_count

        myUtils.write_log('val_test:psnr:{},ssim:{},l1:{},lpips:{}'.format(avg_psnr, avg_ssim, avg_l1, avg_lpips), self.log_path, print_log=True)


        with tqdm(total=len(self.val_from_train_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行测试集验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.g_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_from_train_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    # 拼接图片和mask
                    img_masked = imgs * masks

                    structures_masked = structures * masks

                    context_encode = self.get_context_encode(tags, labels)

                    # 生成图片
                    gen_imgs = self.g_model(img_masked, structures_masked, masks, context_encode)
                    # 补全图片(把mask部分的生成图片和原图拼接)
                    comp_imgs = (imgs * masks) + (gen_imgs * (1 - masks))

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, img_masked, comp_imgs)
                    compare_img_path = os.path.join(self.val_test_from_train_img_save_path, f"{i:0>3d}" + '.jpg')
                    # 保存图片
                    save_image(val_grid, compare_img_path)

                    val_pbar.update(1)





if __name__ == '__main__':
    config = Config()
    trainer = Trainer_Our(config, debug=True)
    # trainer.load_model_last()
    trainer.load_model_epoch(120)
    trainer.val_test()
    # trainer.train()
    #
    # edge_detect = res_skip()
    #
    # edge_detect.load_state_dict(torch.load('../edge_detector/erika.pth'))


