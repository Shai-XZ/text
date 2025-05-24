import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#获取训练的设备（判断是否有GPU）
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#创建模型（基础模型）
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()#展平为以维数据方便全链接层读取
        self.linear_relu_stack = nn.Sequential(
            #输入层或者说全连接层
            nn.Linear(28*28, 512),#28*28就是单张图
            #非线性激活函数
            nn.ReLU(),
            # 隐藏层
            nn.Linear(512, 512),
            # 非线性激活函数
            nn.ReLU(),
            #输出层
            nn.Linear(512, 10),
        )
#前向传播（forward 是模型的灵魂）
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#实例化模型

model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 28, 28, device=device)
#未经归一化的原始预测值（后续需要归一化）
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)#将数随机值规定到0-1之间且总和为1（这10个数）
print(pred_probab)
y_pred = pred_probab.argmax(1)#判断那个数是最大的并保存标签
print(f"Predicted class: {y_pred}")


#清晰理解神经网络中数据从多维输入到特征提取的全过程

#创建一个随机的数据，有3个长宽为28的图像组成
input_image = torch.rand(3,28,28)
print(input_image.size())
#应用展平层（Flatten）
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
#通过线性层（全连接层）
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#非线性激活函数：ReLU作用是将输入值中的所有负数置零，而正数保持不变。
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#模型的一个运行顺序
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#神经网络softmax激活函数
#比如logits张量，他有3个样本每个样本都有10种预测概率，Softmax的作用就是让这10个数加起来为1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)


#模型参数
print(f"Model structure: {model}\n\n")#打印模型的结构

for name, param in model.named_parameters():#显示了权重和偏执（weight，bias）
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")







