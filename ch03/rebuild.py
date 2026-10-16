# ==========================================================================
# 默写练习：从零手写一个 softmax 回归图像分类器
# ==========================================================================
#
# 【需求】
#   用 softmax 回归在 Fashion-MNIST 上训练一个分类器。
#   目标：test acc 达到 80% 以上。
#
# 【规则】
#   1. 不许打开 softmax_scratch.py / softmax_concise.py
#   2. 不许用 nn.Sequential / nn.Linear / nn.CrossEntropyLoss（全部手写）
#   3. 卡住了问 —— 会给你三级提示（L1 看哪儿 / L2 原理 / L3 答案）
#
# 【已知信息】
#   数据已经在 D:\d2l-study\data 里下好了，download=False 直接用
#   图片是 1x28x28 灰度图，10 个类别
#
# 【跑法】
#   & D:\anaconda\envs\torch_env\python.exe d:/d2l-study/ch03/rebuild.py
#
# ==========================================================================
# 文件到此为止。下面你自己写。
# ==========================================================================
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
print("导入成功")
trains = transforms.ToTensor()
minst_train = torchvision.datasets.FashionMNIST(root = r"D:\d2l-study\data", train=True, transform=transforms.ToTensor(), download=False)
minst_test = torchvision.datasets.FashionMNIST(root = r"d:\d2l-study\data", train=False, transform=transforms.ToTensor(), download=False)
print(len(minst_train), len(minst_test))

train_iter = data.DataLoader(minst_train, batch_size=256, shuffle=True)
test_iter = data.DataLoader(minst_test, batch_size=256, shuffle=False)
X, y = next(iter(train_iter))
print(X.shape,y.shape)

W = torch.normal(0, 0.01, size=(784,10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
print(W.shape,b.shape)

def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def net(x):
    o = torch.matmul(x.reshape(-1, W.shape[0]), W) + b
    return o

o = net(X)                 # ③ 用正确的名字
p = softmax(o)
print(p.shape)             # (256, 10)
print(p.sum(dim=1)[:5])    # 应该是 1.0

y_hat = softmax(net(X))
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def evaluate_accuracy(net, data_iter):
    correct = total = 0.0
    with torch.no_grad():
        for X, y in data_iter:
            correct += (softmax(net(X)).argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total
acc = evaluate_accuracy(net, test_iter)
print(acc)

lr = 0.01

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()

def updater(batch_size):
    sgd([W, b], lr, batch_size)




for epoch in range(10):
    total_l, n, correct = 0.0, 0, 0.0
    for X, y in train_iter:
        y_hat = softmax(net(X))                
        l = cross_entropy(y_hat, y)   
        l.sum().backward()
        updater(X.shape[0])
        total_l += l.sum().item()
        n += len(y)
        correct += (y_hat.argmax(dim=1) == y).sum().item()
    test_acc = evaluate_accuracy(net, test_iter)
    print("epoch %2d | train loss %.4f | train acc %.4f | test acc %.4f" % (epoch+1, total_l/n, correct/n, test_acc))