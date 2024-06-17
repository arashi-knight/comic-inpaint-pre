import os
import time
from time import sleep

import torch
from colorama import Fore
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import myUtils
from config import Config
from data import comic_dataloader
from data.dataloader_init import get_dataloader
from other_model.pennet import pennet
from other_model.pennet.core.loss import AdversarialLoss
from myUtils import set_device, get_optimizer, get_optimizer_D, ssim_by_list, psnr_by_list, make_val_grid

class Trainer_PenNet():
    def __init__(self, config: Config, debug=False):
        self.config = config
        self.epoch = 0
        # 迭代次数
        self.iteration = 0

        self.val_img_save_path = config.val_img_save_path_compare
        self.log_path = config.log_path
        self.model_path = config.model_path

        self.best_psnr = 0
        self.best_ssim = 0

        if debug:
            print('debug')

        self.classes, self.train_dataloader, self.test_dataloader, self.val_dataloader = get_dataloader(config)

        # 损失函数
        self.adversarial_loss = set_device(AdversarialLoss(type=self.config.loss_gan_type))
        self.l1_loss = nn.L1Loss()

        # 模型
        self.g_model = set_device(pennet.InpaintGenerator(config=config))
        if self.config.is_gray:
            self.d_model = set_device(
                pennet.Discriminator(in_channels=1, use_sigmoid=self.config.loss_gan_type != 'hinge'))
        else:
            self.d_model = set_device(
                pennet.Discriminator(in_channels=3, use_sigmoid=self.config.loss_gan_type != 'hinge'))

        # 优化器
        self.optimizer_g = get_optimizer(config, self.g_model)
        self.optimizer_d = get_optimizer_D(config, self.d_model)

        # 如果保存路径不存在则创建
        if not os.path.exists(self.val_img_save_path):
            os.makedirs(self.val_img_save_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 加载模型
        self.load_model()

    # 获取学习率
    def get_lr(self, type='G'):
        if type == 'G':
            return self.optimizer_g.param_groups[0]['lr']
        return self.optimizer_d.param_groups[0]['lr']

    # 调整学习率（感觉没啥用）
    def adjust_learning_rate(self):
        decay = 0.1 ** (min(self.iteration, self.config.niter_steady) // self.config.niter)
        new_lr = self.config.learning_rate * decay
        if new_lr != self.get_lr():
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = new_lr

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

    # 训练
    def train_epoch(self):

        # 每个epoch的进度条
        with tqdm(total=len(self.train_dataloader),
                  bar_format=Fore.BLACK + '|{bar:30}|正在进行训练|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as train_pbar:
            for i, (imgs, masks, labels) in enumerate(self.train_dataloader):
                # 设置cuda
                imgs, masks, labels = set_device([imgs, masks, labels])
                # 拼接图片和mask
                img_masked = (imgs * (1 - masks).float()) + masks

                inputs = torch.cat((img_masked, masks), dim=1)

                # 生成图片
                feats, pred_imgs = self.g_model(inputs, masks)
                comp_imgs = (pred_imgs * masks) + (imgs * (1 - masks))

                # 计算损失
                gen_loss = 0
                dis_loss = 0

                # 鉴别器损失
                dis_real_feat = self.d_model(imgs)
                dis_fake_feat = self.d_model(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2

                self.optimizer_d.zero_grad()
                dis_loss.backward()
                self.optimizer_d.step()

                # 生成器对抗损失
                gen_fake_feat = self.d_model(comp_imgs)
                gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False)
                gen_loss += gen_fake_loss * self.config.loss_adversarial_weight

                # 生成器L1损失
                hole_loss = self.l1_loss(masks * pred_imgs, masks * imgs) / torch.mean(masks)
                gen_loss += hole_loss * self.config.loss_hole_weight

                valid_loss = self.l1_loss((1 - masks) * pred_imgs, (1 - masks) * imgs) / torch.mean(1 - masks)
                gen_loss += valid_loss * self.config.loss_valid_weight

                if feats is not None:
                    pyramid_loss = 0
                    for i, feat in enumerate(feats):
                        pyramid_loss += self.l1_loss(feat, F.interpolate(imgs, size=feat.size()[2:4], mode='bilinear',
                                                                         align_corners=True))
                    gen_loss += pyramid_loss * self.config.loss_pyramid_weight

                self.optimizer_g.zero_grad()
                gen_loss.backward()
                self.optimizer_g.step()

                train_pbar.update(1)

    # 训练
    def train(self):
        with tqdm(total=self.config.epochs,
                  bar_format=Fore.MAGENTA + '|{bar:30}|当前epoch:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|{desc}') as epoch_pbar:
            start_time = time.time()
            for epoch in range(self.epoch, self.config.epochs):
                epoch_pbar.update(1)
                self.train_epoch()

                epoch_pbar.display()
                if epoch % self.config.test_interval == 0:
                    psnr, ssim = self.test()
                    # 判断psnr和ssim是否大于之前的最优值
                    if psnr > self.best_psnr and ssim > self.best_ssim:
                        myUtils.write_log('获得最优，当前epoch:{},psnr:{},ssim:{}'.format(epoch, psnr, ssim), self.log_path,
                                        bar=epoch_pbar)
                        self.best_psnr = psnr
                        self.best_ssim = ssim
                        # 保存模型
                        self.save_model()
                        # 保存验证图片
                        self.val()
                    else:
                        myUtils.write_log('未提高，当前epoch:{},psnr:{},ssim:{}'.format(epoch, psnr, ssim), self.log_path,
                                        bar=epoch_pbar)



                # 更新进度条
                # 获取时间数据
                _, time_left, time_end = myUtils.get_time(start_time, epoch, self.config.epochs)
                epoch_pbar.set_description('预计剩余时间:{}|预计结束时间:{}'.format(time_left, time_end))

    # 测试
    def test(self):

        with tqdm(total=len(self.train_dataloader),
                  bar_format=Fore.BLUE + '|{bar:30}|正在进行测试|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as test_pbar:
            avg_psnr = 0
            avg_ssim = 0
            batch_count = 0

            for i, (imgs, masks, labels) in enumerate(self.test_dataloader):
                # 设置cuda
                imgs, masks, labels = set_device([imgs, masks, labels])
                # 拼接图片和mask
                img_masked = (imgs * (1 - masks).float()) + masks

                inputs = torch.cat((img_masked, masks), dim=1)

                # 生成图片
                feats, pred_imgs = self.g_model(inputs, masks)
                comp_imgs = (pred_imgs * masks) + (imgs * (1 - masks))

                # 计算PSNR和SSIM
                # 将图片转换为numpy
                imgs = imgs.detach().cpu().numpy()
                comp_imgs = comp_imgs.detach().cpu().numpy()
                # 计算PSNR和SSIM
                batch_psnr = psnr_by_list(imgs, comp_imgs)
                batch_ssim = ssim_by_list(imgs, comp_imgs)
                # 计算平均PSNR和SSIM
                avg_psnr += batch_psnr
                avg_ssim += batch_ssim
                batch_count += 1
                test_pbar.update(1)

        avg_psnr = avg_psnr / batch_count
        avg_ssim = avg_ssim / batch_count

        return avg_psnr, avg_ssim

    # 验证（保存验证图像）
    def val(self):
        # print('val')
        with tqdm(total=len(self.train_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            for i, (imgs, masks, labels) in enumerate(self.val_dataloader):
                self.iteration += 1
                # 设置cuda
                imgs, masks, labels = set_device([imgs, masks, labels])
                # 拼接图片和mask
                img_masked = (imgs * (1 - masks).float()) + masks

                inputs = torch.cat((img_masked, masks), dim=1)

                # 生成图片
                feats, pred_imgs = self.g_model(inputs, masks)
                comp_imgs = (pred_imgs * masks) + (imgs * (1 - masks))

                # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                val_grid = myUtils.make_val_grid_list([imgs, masks, img_masked, comp_imgs])

                img_path = os.path.join(self.val_img_save_path, f"{i:0>3d}" + '.jpg')
                # 保存图片
                save_image(val_grid, img_path)

                val_pbar.update(1)

    def load_model(self):
        print('load_model')

    def debug(self):
        print('debug')
        print('val')

    def print_model_parm(self):
        print('print_model_parm')
        print('g_total:', sum(p.numel() for p in self.g_model.parameters()) / 1e6)
        print('d_total:', sum(p.numel() for p in self.d_model.parameters()) / 1e6)