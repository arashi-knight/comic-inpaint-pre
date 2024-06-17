import json
import os
import os.path
import random

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from config import Config
from myUtils import ScaleToMinusOneToOne, one_hot, is_binary_tensor, unique_values, unique_values_and_counts, \
    NormalizeMask, ReverseMask, show_tensor


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtname, img_dir, structure_dir, mask_dir, class_to_idx, tag_dir = None):
    images = []
    structures = []
    masks = []
    labels = []
    tags = []

    with open(txtname, 'r',encoding='utf-8') as lines:
        for line in lines:
            class_name = line.split('\\')[0]
            _img = os.path.join(img_dir, line.strip())
            _structure = os.path.join(structure_dir, line.strip())
            _mask = os.path.join(mask_dir, line.strip())

            assert os.path.isfile(_img), _img
            assert os.path.isfile(_mask), _mask
            assert os.path.isfile(_structure), _structure
            images.append(_img)
            structures.append(_structure)
            masks.append(_mask)
            labels.append(class_to_idx[class_name])

            # 如果tag_dir不为空，那么就把tag也加进去
            if tag_dir is not None:
                # .jpg -> .txt
                # line = line.strip().replace('.jpg', '.txt')
                _tag = os.path.join(tag_dir, line.strip().replace('.jpg', '.txt'))
                assert os.path.isfile(_tag), _tag
                tags.append(_tag)

    return images, structures, masks, labels, tags

def make_dataset_v2(txtname, img_dir, structure_dir, mask_dir, mask_json_path, class_to_idx, tag_dir = None):
    images = []
    structures = []
    masks = []
    labels = []
    tags = []

    with open(mask_json_path, 'r', encoding='utf-8') as mask_d:
        mask_dict = json.load(mask_d)

    with open(txtname, 'r',encoding='utf-8') as lines:
        for line in lines:
            class_name = line.split('\\')[0]
            _img = os.path.join(img_dir, line.strip())
            _structure = os.path.join(structure_dir, line.strip())

            # 从mask_dict中获取对应_mask(\替换为/)
            _mask_name = mask_dict[line.strip().replace('\\', '/')]
            _mask = os.path.join(mask_dir, _mask_name)

            assert os.path.isfile(_img), _img
            assert os.path.isfile(_mask), _mask
            assert os.path.isfile(_structure), _structure
            images.append(_img)
            structures.append(_structure)
            masks.append(_mask)
            labels.append(class_to_idx[class_name])

            # 如果tag_dir不为空，那么就把tag也加进去
            if tag_dir is not None:
                # .jpg -> .txt
                # line = line.strip().replace('.jpg', '.txt')
                _tag = os.path.join(tag_dir, line.strip().replace('.jpg', '.txt'))
                assert os.path.isfile(_tag), _tag
                tags.append(_tag)

    return images, structures, masks, labels, tags


class ComicDataloader(data.Dataset):
    def __init__(self, config: Config, img_transform = None, mask_transform = None, type = 'train'):
        # 读取类别
        classes, class_to_idx = find_classes(config.img_path)
        self.classes = classes
        self.classes_num = len(classes)
        self.class_to_idx = class_to_idx
        self.type = type
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_size = config.img_size
        if type == 'train':
            text_name = config.train_txt
        elif type == 'test':
            text_name = config.test_txt
        elif type == 'val':
            text_name = config.val_txt
        elif type == 'val_from_train':
            text_name = config.val_from_train_txt
        else:
            raise ValueError('type must be train or test')

        self.is_gray = config.is_gray

        if config.mask_mode == 1:
            self.images, self.structures, self.masks, self.labels, self.tags = make_dataset(text_name, config.img_path, config.structure_path, config.mask_path, class_to_idx, config.tag_path)
        elif config.mask_mode == 2:
            self.images, self.structures, self.masks, self.labels, self.tags = make_dataset_v2(text_name, config.img_path, config.structure_path, config.mask_json_data_path, config.mask_json_path, class_to_idx, config.tag_path)

        assert (len(self.images) == len(self.labels) == len(self.masks))

    def __getitem__(self, index):
        # 读取图片
        if self.is_gray:
            _img = Image.open(self.images[index]).convert('L')
        else:
            _img = Image.open(self.images[index]).convert('RGB')
        # 读取结构和mask
        _structure = Image.open(self.structures[index]).convert('L')
        _mask = Image.open(self.masks[index]).convert('L')
        # 读取分类（漫画名差分）
        _label = self.labels[index]
        # 转为one-hot编码
        _label = one_hot(_label, self.classes_num)
        # 读取tag为字符串(txt文件)
        if len(self.tags) > 0:
            _tag = open(self.tags[index], 'r', encoding='utf-8').read().strip()
        else:
            _tag = None

        # # 手工实现transform
        #
        # # resize
        # _img = _img.resize((self.img_size, self.img_size))
        # _structure = _structure.resize((self.img_size, self.img_size))
        # _mask = _mask.resize((self.img_size, self.img_size))
        #
        # # 概率性的水平翻转
        # if random.random() > 0.5:
        #     _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
        #     _structure = _structure.transpose(Image.FLIP_LEFT_RIGHT)
        #     _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # 转为tensor
        # _img = transforms.ToTensor()(_img)
        # _structure = transforms.ToTensor()(_structure)
        # _mask = transforms.ToTensor()(_mask)



        if self.img_transform is not None:
            _img = self.img_transform(_img)
            _structure = self.img_transform(_structure)
        else:
            _img = transforms.ToTensor()(_img)
            _structure = transforms.ToTensor()(_structure)
        if self.mask_transform is not None:
            _mask = self.mask_transform(_mask)
        else:
            _mask = transforms.ToTensor()(_mask)


        return _img, _structure, _mask, _label, _tag

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self, config: Config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.Resize(config.img_size),
            # transforms.RandomResizedCrop(config.img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            # ScaleToMinusOneToOne(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(config.img_size),
            # transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            # ScaleToMinusOneToOne(),
        ])

        transform_mask = transforms.Compose([
            transforms.Resize(config.img_size),
            # transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            NormalizeMask(threshold=0.2),
            ReverseMask()
        ])

        trainset = ComicDataloader(config, transform_train, transform_mask, type='train')
        testset = ComicDataloader(config, transform_test, transform_mask, type='test')
        valsset = ComicDataloader(config, transform_test, transform_mask, type='val')
        val_from_trainset = ComicDataloader(config, transform_test, transform_mask, type='val_from_train')

        train_kwargs = {'shuffle': True,'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        val_kwargs = {'shuffle': False,'num_workers': config.num_workers, 'pin_memory': config.pin_memory}

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
        config.batch_size, **train_kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
        config.batch_size, **train_kwargs)
        valloader = torch.utils.data.DataLoader(valsset, batch_size=
        config.batch_size, **val_kwargs)
        val_from_trainloader = torch.utils.data.DataLoader(val_from_trainset, batch_size=
        config.batch_size, **val_kwargs)

        self.classes = trainset.classes
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.val_from_trainloader = val_from_trainloader

    def getloader(self):
        return self.classes, self.trainloader, self.testloader, self.valloader, self.val_from_trainloader

def getloader(config: Config):
    return Dataloder(config).getloader()



# 测试数据集
def test_dataloader():
    config = Config()
    config.mask_mode = 2
    config.batch_size = 1
    dataloader = Dataloder(config)
    classes, trainloader, testloader, valloader, val_from_train_loader = dataloader.getloader()
    print(classes)
    for i, (imgs, structures, masks, labels, tags) in enumerate(trainloader):
        print('imgs:', type(imgs), imgs.shape)
        print('structures:', type(structures), structures.shape)
        print('masks:', type(masks), masks.shape)
        print('labels:', type(labels), labels)
        print('tags:', type(tags), tags)

        show_tensor(masks)


        # 判断mask是否只包含0和1
        print('mask是否只包含0和1:', is_binary_tensor(masks))
        unique, count = unique_values_and_counts(masks)
        for j in range(len(unique)):
            print('mask中{}的数量为{}'.format(unique[j], count[j]))

        # 循环5次
        if i > 5:
            break

if __name__ == '__main__':
    test_dataloader()

    # x = torch.randn(1,1,26)
    # y = torch.randn(1,77,768)
    #
    # x = x.expand(-1,77,-1)
    # print(x.shape)
    #
    # z = torch.cat([x,y], dim=2)
    #
    # print(z.shape)
