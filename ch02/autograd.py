import torch

x = torch.arange(4.0)
#x.requires_grad_(True)
y = 2*torch.dot(x,x)
print("y = ",y)
y.backward()
print("x.grad = ",x.grad)
print(x.grad == 4 * x)