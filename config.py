import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# 将项目地址添加到模块搜索路径的列表中
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 转化为相对地址
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class Config():
    # 本次训练的编号
    train_id: int = 2
    # 输入图像大小
    img_size: int = 256
    # 训练轮次
    epochs: int = 20
    # batch_size
    batch_size: int = 4
    # 输入通道数
    input_channels: int = 1
    # 输出通道数
    output_channels: int = 1
    # context信息的通道数，768为clip的输出通道数，26为tag的one-hot编码通道数
    context_channels: int = 768+26
    # encode和decode块的基础通道数
    base_block_num: int = 32

    # clip版本
    clip_version:str = 'openai/clip-vit-large-patch14'
    # clip_layer的参数('last','pooled','hidden','all')
    clip_layer: str = 'hidden'
    # cilp_skipping的层数
    clip_skipping: int = -2
    # 数据归一化的范围
    data_range: int = 2

    # data_base_path = r'F:\zl\data\comic_dataset'
    # 本机数据集基准地址
    data_base_path = r'E:\CODE\dataset\Comic\comic_dataset'
    # ng-1数据集基准地址
    # data_base_path = r'F:\hzl\data\comic_dataset'
    # 3070数据集基准地址
    # data_base_path = r'/home/boss_wong/HZL/data/comic_dataset'

    # mask模式（1；一一对应，2；根据json读取）
    mask_mode: str = 2

    # mask_json地址
    mask_json_path: str = os.path.join(data_base_path, 'comic_mask_Irregular_json', 'border', '10_20.json')
    # mask_json_data地址
    mask_json_data_path: str = os.path.join(data_base_path, 'comic_mask_Irregular')



    # 图像地址
    # img_path: str = data_base_path +  r'\comic_label_slice_resize_256'
    img_path: str = os.path.join(data_base_path, 'comic_label_slice_resize_256')
    # mask地址
    # mask_path: str = data_base_path + r'\comic_label_slice_reszie_mask\comic_mask_256_2_100_50'
    mask_path: str = os.path.join(data_base_path, 'comic_label_slice_reszie_mask', 'comic_mask_256_2_100_50')
    # structure地址
    # structure_path: str = data_base_path + r'\comic_label_slice_resize_structure'
    structure_path: str = os.path.join(data_base_path, 'comic_label_slice_resize_structure')
    # tag地址
    # tag_path: str = data_base_path + r'\comic_tag_noValue'
    tag_path: str = os.path.join(data_base_path, 'comic_tag_noValue')

    # 训练集、测试集、验证集的txt地址编号
    train_txt_num: int = 1

    # 训练集txt地址
    # train_txt: str = data_base_path + r'\train_test_label\train{}.txt'.format(train_txt_num)
    train_txt: str = os.path.join(data_base_path, 'train_test_label', 'train{}.txt'.format(train_txt_num))
    # 测试集txt地址
    # test_txt: str = data_base_path + r'\train_test_label\test{}.txt'.format(train_txt_num)
    test_txt: str = os.path.join(data_base_path, 'train_test_label', 'test{}.txt'.format(train_txt_num))
    # 验证集txt地址
    # val_txt: str = data_base_path + r'\train_test_label\val{}.txt'.format(train_txt_num)
    val_txt: str = os.path.join(data_base_path, 'train_test_label', 'val{}.txt'.format(train_txt_num))
    # 从训练集中划分的验证集
    val_from_train_txt: str = os.path.join(data_base_path, 'train_test_label', 'val_from_train{}.txt'.format(train_txt_num))


    # svae模型预训练模型地址
    svae_model_path: str = os.path.join(data_base_path, 'pretrained_model', 'ScreenVAE')
    # 边缘检测模型地址
    edge_model_path: str = os.path.join(data_base_path, 'pretrained_model', 'erika.pth')

    # 是否使用灰度图
    is_gray: bool = True
    # 数据集参数
    num_workers: int = 0
    pin_memory: bool = False

    # 是否cuda
    is_cuda: bool = True

    dataset_name: str = 'comic'

    # 损失函数类型，可选：["nsgan", "lsgan", "hinge"]
    loss_gan_type:str = "hinge"
    # 损失函数权重
    loss_hole_weight = 6.0
    loss_valid_weight = 1.0
    loss_l1_weight = 1.0
    loss_pyramid_weight = 0.5
    loss_adversarial_weight = 0.1
    loss_perceptual_weight = 0.05
    loss_style_weight = 250.0

    loss_texture_weight = 0.
    loss_mask_edge_weight = 0.
    loss_structure_weight = 10
    loss_bin_weight = 0.

    # 优化器类型，可选：["sgd", "adam"]
    optimizer_type = "adam"
    # 学习率
    learning_rate = 3e-4
    # 鉴别器到生成器的学习率比例
    discriminator_lr_factor = 0.1

    # adam优化器参数
    adam_betas = (0.9, 0.999)
    # 学习率衰减参数
    niter_steady = 1e4
    niter = 10e4

    # sgd优化器参数
    sgd_momentum = 0.9

    # 是否使用学习率衰减
    lr_decay = False
    # 学习率衰减策略
    lr_decay_policy = "step"
    # setp学习率衰减间隔
    lr_decay_epoch = epochs//2
    # step学习率衰减幅度
    step_lr_decay_factor = 0.5

    # 模型测试间隔
    test_interval:int = 1
    # 模型保存间隔
    save_interval:int = 20

    # 结果保存路径
    result_path: str = os.path.join(ROOT, 'results', str(train_id))

    # 模型保存路径
    model_path: str = os.path.join(result_path, 'checkpoint')
    # val结果保存路径-对比图
    val_img_save_path_compare: str = os.path.join(result_path, 'val_img', 'compare')
    # val结果保存路径-单图
    val_img_save_path_single: str = os.path.join(result_path, 'val_img', 'single')
    # 最优val结果保存路径
    best_val_img_save_path_compare: str = os.path.join(result_path, 'best_val_img', 'compare')
    best_val_img_save_path_single: str = os.path.join(result_path, 'best_val_img', 'single')

    # val_from_train结果保存路径
    val_from_train_img_save_path_compare: str = os.path.join(result_path, 'val_from_train_img')

    # log保存路径
    log_path: str = os.path.join(result_path, 'log.txt')
    # 用于调试的log保存路径
    debug_log_path: str = os.path.join(result_path, 'debug_log.txt')
    # 用于保存config的路径
    config_path: str = os.path.join(result_path, 'config.json')
    # psnr折线图保存路径
    psnr_img_save_path: str = os.path.join(result_path, 'psnr_img.png')
    # ssim折线图保存路径
    ssim_img_save_path: str = os.path.join(result_path, 'ssim_img.png')
    # lpips折线图保存路径
    lpips_img_save_path: str = os.path.join(result_path, 'lpips_img.png')
    # test_l1_loss折线图保存路径
    test_l1_loss_img_save_path: str = os.path.join(result_path, 'test_l1_loss_img.png')
    # train_l1_loss折线图保存路径
    train_l1_loss_img_save_path: str = os.path.join(result_path, 'train_l1_loss_img.png')


    def to_json(self):
        return {
            'train_id': self.train_id,
            'img_size': self.img_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'context_channels': self.context_channels,
            'base_block_num': self.base_block_num,
            'clip_version': self.clip_version,
            'clip_layer': self.clip_layer,
            'clip_skipping': self.clip_skipping,
            'data_range': self.data_range,
            'data_base_path': self.data_base_path,
            'mask_mode': self.mask_mode,
            'mask_json_path': self.mask_json_path,
            'mask_json_data_path': self.mask_json_data_path,
            'img_path': self.img_path,
            'mask_path': self.mask_path,
            'structure_path': self.structure_path,
            'tag_path': self.tag_path,
            'train_txt_num': self.train_txt_num,
            'train_txt': self.train_txt,
            'test_txt': self.test_txt,
            'val_txt': self.val_txt,
            'val_from_train_txt': self.val_from_train_txt,
            'is_gray': self.is_gray,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'is_cuda': self.is_cuda,
            'dataset_name': self.dataset_name,
            'loss_gan_type': self.loss_gan_type,
            'loss_hole_weight': self.loss_hole_weight,
            'loss_valid_weight': self.loss_valid_weight,
            'loss_l1_weight': self.loss_l1_weight,
            'loss_pyramid_weight': self.loss_pyramid_weight,
            'loss_adversarial_weight': self.loss_adversarial_weight,
            'loss_perceptual_weight': self.loss_perceptual_weight,
            'loss_style_weight': self.loss_style_weight,
            'loss_texture_weight': self.loss_texture_weight,
            'loss_mask_edge_weight': self.loss_mask_edge_weight,
            'loss_structure_weight': self.loss_structure_weight,
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'discriminator_lr_factor': self.discriminator_lr_factor,
            'adam_betas': self.adam_betas,
            'niter_steady': self.niter_steady,
            'niter': self.niter,
            'sgd_momentum': self.sgd_momentum,
            'lr_decay': self.lr_decay,
            'lr_decay_policy': self.lr_decay_policy,
            'lr_decay_epoch': self.lr_decay_epoch,
            'step_lr_decay_factor': self.step_lr_decay_factor,
            'test_interval': self.test_interval,
            'save_interval': self.save_interval,
            'result_path': self.result_path,
            'model_path': self.model_path,
            'val_img_save_path_compare': self.val_img_save_path_compare,
            'val_img_save_path_single': self.val_img_save_path_single,
            'best_val_img_save_path_compare': self.best_val_img_save_path_compare,
            'best_val_img_save_path_single': self.best_val_img_save_path_single,
            'val_from_train_img_save_path_compare': self.val_from_train_img_save_path_compare,
            'log_path': self.log_path,
            'debug_log_path': self.debug_log_path,
            'config_path': self.config_path,
            'psnr_img_save_path': self.psnr_img_save_path,
            'ssim_img_save_path': self.ssim_img_save_path,
            'lpips_img_save_path': self.lpips_img_save_path,
            'test_l1_loss_img_save_path': self.test_l1_loss_img_save_path,
            'train_l1_loss_img_save_path': self.train_l1_loss_img_save_path
        }

def update_config(config: Config, **kwargs):

    # 更新训练编号
    if 'train_id' in kwargs:
        # 本次训练的编号
        config.train_id = kwargs['train_id']
        # 结果保存路径
        config.result_path = os.path.join(ROOT, 'results', str(config.train_id))
        # 模型保存路径
        config.model_path = os.path.join(config.result_path, 'checkpoint')
        # val结果保存路径
        config.val_img_save_path_compare = os.path.join(config.result_path, 'val_img')
        # log保存路径
        config.log_path = os.path.join(config.result_path, 'log.txt')
        # psnr折线图保存路径
        config.psnr_img_save_path = os.path.join(config.result_path, 'psnr_img.png')
        # ssim折线图保存路径
        config.ssim_img_save_path = os.path.join(config.result_path, 'ssim_img.png')
        # l1折线图
        config.test_l1_loss_img_save_path = os.path.join(config.result_path, 'test_l1_loss_img.png')
        config.train_l1_loss_img_save_path = os.path.join(config.result_path, 'train_l1_loss_img.png')

    # 更新学习率
    if 'learning_rate' in kwargs:
        config.learning_rate = kwargs['learning_rate']

    # 更新训练集
    if 'train_txt_num' in kwargs:
        config.train_txt_num = kwargs['train_txt_num']
        config.train_txt = os.path.join(config.data_base_path, 'train_test_label', 'train{}.txt'.format(config.train_txt_num))
        config.test_txt = os.path.join(config.data_base_path, 'train_test_label', 'test{}.txt'.format(config.train_txt_num))
        config.val_txt = os.path.join(config.data_base_path, 'train_test_label', 'val{}.txt'.format(config.train_txt_num))

    # 更新clip_skipping跳过层数
    if 'clip_skipping' in kwargs:
        config.clip_skipping = kwargs['clip_skipping']








class Config_pennet(Config):
    # 输入图像大小
    img_size: int = 128


if __name__ == '__main__':
    config = Config()
    print(config.to_json())
