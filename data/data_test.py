import torch
from torch.utils.data import DataLoader, Dataset

# 创建一个自定义的数据集，这是一个示例数据集
class CustomDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        return self.data1[index], self.data2[index]

# 创建数据集示例
data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2 = [12, 2, 3, 14, 5, 6, 7, 8, 9, 10]
dataset = CustomDataset(data1, data2)

# 创建 DataLoader 并设置 shuffle=True 来随机打乱数据
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 遍历 DataLoader
for data in dataloader:
    print('data1:', data[0], 'data2:', data[1])