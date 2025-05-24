#自动微分就是训练模型的过程（定义模型和优化器，进行梯度清零和前向反向传播最后更新参数）
import torch
#实现了二分类逻辑回归模型
x = torch.ones(5)  # 输入张量
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)#requires_grad=True，，启用梯度跟踪
z = torch.matmul(x, w)+b#线性变换x*w+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)#计算二元交叉熵损失

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


#计算梯度
loss.backward()
print(w.grad)#权重
print(b.grad)#偏置

#禁用渐变跟踪（作用就是减少内存的占用，提升推理速度）
z = torch.matmul(x, w)+b
print(z.requires_grad)
#禁用的两个方法
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
#使用在Tensor上
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

#*可选张量梯度和雅可比积（先过后续在去扩展学习）*
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")