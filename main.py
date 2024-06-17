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
from trainer.trainer_our import Trainer_Our
from trainer.trainer_pennet import Trainer_PenNet



if __name__ == '__main__':
    config = Config()
    # 如果训练编号已经存在，输入回车覆盖，否则退出
    if os.path.exists(os.path.join(config.result_path)):
        print(Fore.RED + '训练编号已经存在，是否覆盖？(y/n)')
        if input() != 'y':
            exit()

    myUtils.save_config(config, config.config_path)
    trainer = Trainer_Our(config)
    trainer.val()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
