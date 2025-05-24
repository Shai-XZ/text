import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#下载数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
#定义超参数
learning_rate = 1e-3#学习率
batch_size = 64 #每一批训练多少数据
epochs = 10  #迭代多少次

#加载数据集
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


#创建模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),#输入层
            nn.ReLU(),#激活函数
            nn.Linear(512, 512),#隐藏层
            nn.ReLU(),#激活函数
            nn.Linear(512, 10),#输出层
        )
#forward进行前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#初始化模型
model = NeuralNetwork()


# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#训练模型
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)#获取训练集的样本大小

    model.train()#将模型设置为训练模式
    for batch, (X, y) in enumerate(dataloader):#x是数据，Y是对应的标签
        # 前向传播计算损失值
        pred = model(X)#对X进行预测
        loss = loss_fn(pred, y)#计算预测的损失，用到了loss_fn损失函数

        #反向传播优化参数
        loss.backward()#计算梯度用到了backward函数
        optimizer.step()#更新参数
        optimizer.zero_grad()#梯度清零
#打印loos，和current，size值
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#测试模型
def test_loop(dataloader, model, loss_fn):

    model.eval()#将模型设置为测试模式（禁用训练专用层）
    size = len(dataloader.dataset)#获取测试集的样本大小
    num_batches = len(dataloader)#测试集的批次
    test_loss, correct = 0, 0#初始化累计损失和正确预测数

#禁用梯度计算
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 迭代损失值
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")