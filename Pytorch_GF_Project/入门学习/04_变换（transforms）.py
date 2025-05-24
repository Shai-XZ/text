
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


#下载数据集
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
# 自定义变换函数（Lambda）
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#使用 Lambda 变换将标签转换为 one-hot 编码
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

