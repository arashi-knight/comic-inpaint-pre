import os
import random


def split_dataset(input_folder, output_train_txt, output_test_txt, split_ratio=0.8):
    if not os.path.exists(input_folder):
        print(f"目标文件夹 {input_folder} 不存在")
        return

    # 获取目标文件夹下的所有图像文件
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # 随机打乱图像文件列表
    random.shuffle(image_files)

    # 计算划分训练集和测试集的边界索引
    split_index = int(len(image_files) * split_ratio)

    # 将图像文件分为训练集和测试集
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # 将训练集和测试集的相对路径分别写入txt文档
    with open(output_train_txt, 'a') as train_file:
        for file in train_files:
            print('file', file)
            relative_path = os.path.relpath(file, input_folder)
            train_file.write(relative_path + '\n')

    with open(output_test_txt, 'a') as test_file:
        for file in test_files:
            relative_path = os.path.relpath(file, input_folder)
            test_file.write(relative_path + '\n')

    print(f"成功将图像数据按照 {split_ratio*100}% 的比例划分为训练集和测试集")
    print(f"训练集路径已写入 {output_train_txt}")
    print(f"测试集路径已写入 {output_test_txt}")

def split_dataset_with_subfolders_train_test(target_folder, output_train_txt, output_test_txt, split_ratio = 0.8):
    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在")
        return
    # 获取目标文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]

    # 创建训练集和测试集文件的路径
    train_txt = open(output_train_txt, "w", encoding='utf-8')
    test_txt = open(output_test_txt, "w", encoding='utf-8')

    for folder in subfolders:
        print(f"正在处理 {folder} 文件夹")
        folder_path = os.path.join(target_folder, folder)
        files = os.listdir(folder_path)
        random.shuffle(files)

        # 计算训练集和测试集的切分点
        total_files = len(files)
        train_split = int(total_files * split_ratio)
        test_split = int(total_files * (1 - split_ratio))

        # 写入训练集文件路径
        for file in files[:train_split]:
            train_txt.write(os.path.join(folder, file) + "\n")

        # 写入测试集文件路径
        for file in files[train_split:train_split + test_split]:
            test_txt.write(os.path.join(folder, file) + "\n")

    # 关闭文件
    train_txt.close()
    test_txt.close()

def split_dataset_with_subfolders_train_test_val(target_folder, output_train_txt, output_test_txt, output_val_txt, train_ratio = 0.8, test_ratio = 0.2):

    # 如果比例不合法，直接返回
    if train_ratio + test_ratio > 1:
        print("训练集和测试集的比例之和不能大于1")
        return

    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在")
        return
    # 获取目标文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]

    # 创建训练集和测试集文件的路径
    train_txt = open(output_train_txt, "w", encoding='utf-8')
    test_txt = open(output_test_txt, "w", encoding='utf-8')
    val_txt = open(output_val_txt, "w", encoding='utf-8')

    for folder in subfolders:
        print(f"正在处理 {folder} 文件夹")
        folder_path = os.path.join(target_folder, folder)
        files = os.listdir(folder_path)
        random.shuffle(files)

        # 计算训练集和测试集的切分点
        total_files = len(files)
        train_split = int(total_files * train_ratio)
        test_split = int(total_files * test_ratio)

        # 写入训练集文件路径
        for file in files[:train_split]:
            train_txt.write(os.path.join(folder, file) + "\n")

        # 写入测试集文件路径
        for file in files[train_split:train_split + test_split]:
            test_txt.write(os.path.join(folder, file) + "\n")

        # 写入验证集文件路径
        for file in files[train_split + test_split:]:
            val_txt.write(os.path.join(folder, file) + "\n")

    # 关闭文件
    train_txt.close()
    test_txt.close()
    val_txt.close()

# 生成训练集和测试集和验证集，这个版本的验证集包含在测试集中
def split_dataset_with_subfolders_train_test_val_v2(target_folder, output_train_txt, output_test_txt, output_val_txt, train_ratio = 0.8, test_ratio = 0.2, val_ratio = 0.1):

    # 如果比例不合法，直接返回
    assert train_ratio + test_ratio <= 1, "训练集和测试集的比例不能大于1"

    # val的比例不能大于test的比例
    assert val_ratio < test_ratio, "val的比例不能大于test的比例"

    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在")
        return
    # 获取目标文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]

    # 创建训练集和测试集文件的路径
    train_txt = open(output_train_txt, "w", encoding='utf-8')
    test_txt = open(output_test_txt, "w", encoding='utf-8')
    val_txt = open(output_val_txt, "w", encoding='utf-8')

    total_train = 0
    total_test = 0
    total_val = 0

    for folder in subfolders:
        print(f"正在处理 {folder} 文件夹")
        folder_path = os.path.join(target_folder, folder)
        files = os.listdir(folder_path)
        random.shuffle(files)

        # 计算训练集和测试集的切分点
        total_files = len(files)
        train_split = int(total_files * train_ratio)
        test_split = int(total_files * test_ratio)
        val_split = int(total_files * val_ratio)

        total_train += train_split
        total_test += test_split
        total_val += val_split

        # 写入训练集文件路径
        for file in files[:train_split]:
            train_txt.write(os.path.join(folder, file) + "\n")

        # 写入测试集文件路径
        for file in files[train_split:train_split + test_split]:
            test_txt.write(os.path.join(folder, file) + "\n")

        # 写入验证集文件路径
        for file in files[train_split + (test_split-val_split):train_split + test_split]:
            val_txt.write(os.path.join(folder, file) + "\n")

    print('成功划分，训练集数据量为：', total_train, '测试集数据量为：', total_test, '验证集数据量为：', total_val)

    # 关闭文件
    train_txt.close()
    test_txt.close()
    val_txt.close()

def split_dataset_with_subfolders_train_test_val_v3(target_folder, output_train_txt, output_test_txt, output_val_txt, output_val_from_train_txt,
                                                    train_ratio = 0.8, test_ratio = 0.2, val_ratio = 0.1, val_from_train_ratio = 0.1):
    """
    生成训练集和测试集和验证集，这个版本的验证集包含在测试集和训练集中
    :param target_folder:
    :param output_train_txt:
    :param output_test_txt:
    :param output_val_txt:
    :param output_val_from_train_txt:
    :param train_ratio:
    :param test_ratio:
    :param val_ratio:
    :param val_from_train_ratio:
    :return:
    """

    # 如果比例不合法，直接返回
    assert train_ratio + test_ratio <= 1, "训练集和测试集的比例不能大于1"

    # val的比例不能大于test的比例
    assert val_ratio < test_ratio, "val的比例不能大于test的比例"

    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在")
        return
    # 获取目标文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]

    # 创建训练集和测试集文件的路径
    train_txt = open(output_train_txt, "w", encoding='utf-8')
    test_txt = open(output_test_txt, "w", encoding='utf-8')
    val_txt = open(output_val_txt, "w", encoding='utf-8')
    val_from_train_txt = open(output_val_from_train_txt, "w", encoding='utf-8')

    total_train = 0
    total_test = 0
    total_val = 0

    for folder in subfolders:
        print(f"正在处理 {folder} 文件夹")
        folder_path = os.path.join(target_folder, folder)
        files = os.listdir(folder_path)
        random.shuffle(files)

        # 计算训练集和测试集的切分点
        total_files = len(files)
        train_split = int(total_files * train_ratio)
        test_split = int(total_files * test_ratio)
        val_split = int(total_files * val_ratio)
        val_from_train_split = int(total_files * val_from_train_ratio)

        total_train += train_split
        total_test += test_split
        total_val += val_split

        # 写入训练集文件路径
        for file in files[:train_split]:
            train_txt.write(os.path.join(folder, file) + "\n")

        # 写入测试集文件路径
        for file in files[train_split:train_split + test_split]:
            test_txt.write(os.path.join(folder, file) + "\n")

        # 写入验证集文件路径
        for file in files[train_split + (test_split-val_split):train_split + test_split]:
            val_txt.write(os.path.join(folder, file) + "\n")

        # 写入val_from_train_txt
        for file in files[:val_from_train_split]:
            val_from_train_txt.write(os.path.join(folder, file) + "\n")

    print('成功划分，训练集数据量为：', total_train, '测试集数据量为：', total_test, '验证集数据量为：', total_val)

    # 关闭文件
    train_txt.close()
    test_txt.close()
    val_txt.close()
if __name__ == '__main__':
    pass
    path = 'E:\CODE\dataset\Comic\comic_label_slice_resize\comic_label_slice_resize_256'

    # 创建训练集和测试集的txt文件，数据为90%训练集，10%测试集
    for i in range(1,4):
        train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\train{i}.txt'
        test_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\test{i}.txt'
        val_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val{i}.txt'
        val_from_train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val_from_train{i}.txt'

        split_dataset_with_subfolders_train_test_val_v3(path, train_txt, test_txt, val_txt, val_from_train_txt, train_ratio=0.9, test_ratio=0.1, val_ratio=0.01, val_from_train_ratio=0.01)

    # 创建训练集和测试集的txt文件，数据为40%训练集，5%测试集
    for i in range(4,7):
        train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\train{i}.txt'
        test_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\test{i}.txt'
        val_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val{i}.txt'
        val_from_train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val_from_train{i}.txt'

        split_dataset_with_subfolders_train_test_val_v3(path, train_txt, test_txt, val_txt, val_from_train_txt, train_ratio=0.4, test_ratio=0.05, val_ratio=0.01, val_from_train_ratio=0.01)

    # 创建训练集和测试集的txt文件，数据为20%训练集，5%测试集
    for i in range(7,10):
        train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\train{i}.txt'
        test_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\test{i}.txt'
        val_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val{i}.txt'
        val_from_train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val_from_train{i}.txt'

        split_dataset_with_subfolders_train_test_val_v3(path, train_txt, test_txt, val_txt, val_from_train_txt, train_ratio=0.2, test_ratio=0.05, val_ratio=0.01, val_from_train_ratio=0.01)

    for i in range(10,13):
        train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\train{i}.txt'
        test_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\test{i}.txt'
        val_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val{i}.txt'
        val_from_train_txt = f'E:\CODE\dataset\Comic\\train_test_label_en\\val_from_train{i}.txt'

        split_dataset_with_subfolders_train_test_val_v3(path, train_txt, test_txt, val_txt, val_from_train_txt, train_ratio=0.02, test_ratio=0.02, val_ratio=0.01, val_from_train_ratio=0.01)

