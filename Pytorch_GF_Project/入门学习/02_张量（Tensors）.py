import torch
import numpy as np

#创建张量
data = [[1,2,3],[4,5,6],[7,8,9]]
x_data = torch.tensor(data)
print(x_data)
#修改张量的数据类型
x_data = torch.tensor(data).to(torch.float)

#使用numpy创建张量
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


#初始化和覆盖初始化张量
x_ones = torch.ones_like(x_data) # 保留 x_data 的形态包括：数据类型，形状（3x3）
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data,dtype=torch.float) # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")


#指定存储的值
shape = (2,3,)
rand_tensor = torch.rand(shape)#随机值
ones_tensor = torch.ones(shape)#都为1
zeros_tensor = torch.zeros(shape)#都为0

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#查看Tensor的属性（使用的形状，数据类型，和运行设备)
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")#形状
print(f"Datatype of tensor: {tensor.dtype}")#数据类型
print(f"Device tensor is stored on: {tensor.device}")#使用CPU或GPU

# 将张量移动到GPU
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

#张量的索引和切片
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
#连接张量（cat方法）
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#矩阵乘法
y1 = tensor @ tensor.T          # 使用 @ 运算符
y2 = tensor.matmul(tensor.T)    # 使用张量的 .matmul() 方法

y3 = torch.rand_like(y1)        # 预分配内存
torch.matmul(tensor, tensor.T, out=y3)  # 使用 torch.matmul() 并指定输出

#逐元素乘积
z1 = tensor * tensor          # 使用 * 运算符
z2 = tensor.mul(tensor)       # 使用张量的 .mul() 方法

z3 = torch.rand_like(tensor)  # 预分配内存
torch.mul(tensor, tensor, out=z3)  # 使用 torch.mul() 并指定输出

#把张量里的数都加起来，使用item（）把其换为 Python 数值使用
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#就地调用（每个元素都加5）
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


#数据共享

#Tensor 到 NumPy 数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#共享的体现，只要更改一个数，两个数都会变
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy到Tensor数组
n = np.ones(5)
t = torch.from_numpy(n)

#共享的体现，只要更改一个数，两个数都会变
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")