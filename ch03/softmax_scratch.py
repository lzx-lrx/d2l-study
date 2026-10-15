# D2L 3.6 softmax 回归的从零开始实现
# 目标：不用 nn.Sequential / nn.CrossEntropyLoss，全部手搓一个图像分类器

import time
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

torch.manual_seed(0)          # 固定随机种子


# ============================================================
# 1. 数据
# ============================================================
trans = transforms.ToTensor()
root = r"D:\d2l-study\data"
mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False,
                                               transform=trans, download=False)

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)


# ============================================================
# 2. 参数
# ============================================================
num_inputs, num_outputs = 784, 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

print("W.shape =", tuple(W.shape), "   <- (输入维度, 类别数)")
print("b.shape =", tuple(b.shape))
print("参数总数 =", W.numel() + b.numel())


# ============================================================
# 3. softmax（你 softmax_basic.py 写过，直接给）
# ============================================================
def softmax(X):
    X_exp = torch.exp(X)
    return X_exp / X_exp.sum(1, keepdim=True)
    # sum(1) = 对「每一行」求和。一行 = 一个样本的 10 个分数


# ============================================================
# 4. 模型
# ============================================================
# TODO(你) 1/2：前向 —— 拉平 -> 矩阵乘 -> 加偏置 -> softmax
#   X 的形状是 (batch, 1, 28, 28)，必须拉平成 (batch, 784)
#   hint: X.reshape((-1, W.shape[0]))     <- -1 让 PyTorch 自己算 batch
#   hint: torch.matmul(拉平后的 X, W) + b
#   hint: 外面再套一层 softmax(...)
def net(X):
    X_flat = X.reshape((-1, W.shape[0]))
    o = torch.matmul(X_flat, W) + b
    a = softmax(o)
    return a


# ============================================================
# 5. 损失（你 cross_entropy.py 写过，直接给）
# ============================================================
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
    # 注意：返回的是「向量」，一个样本一个损失，所以后面要 .sum()


# ============================================================
# 6. 精度
# ============================================================
# TODO(你) 2/2：算这一批里猜对了多少个
#   三步：
#     1) 把每个样本「概率最大的类别」取出来：y_hat.argmax(axis=1)
#        （先判断 y_hat 是不是二维的、第二维 > 1）
#     2) 和真实标签比较：y_hat.type(y.dtype) == y      -> 得到 True/False
#     3) 数一下 True 的个数，返回 float
#   hint: cmp.type(y.dtype).sum()
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
         y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())


# ============================================================
# 7. 在测试集上评估
# ============================================================
def evaluate_accuracy(net, data_iter):
    correct, total = 0.0, 0.0
    with torch.no_grad():               # 评估不需要梯度
        for X, y in data_iter:
            correct += accuracy(net(X), y)
            total += y.numel()
    return correct / total


# ============================================================
# 8. 优化器
# ============================================================
lr = 0.1


def sgd(params, lr, bs):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / bs
            p.grad.zero_()


def updater(bs):
    sgd([W, b], lr, bs)
    # ⚠️ 必须把 [W, b] 当参数传进去！
    #    如果直接在这里写 `W -= ...`，W 会变成 updater 的「局部变量」，
    #    报 UnboundLocalError。这是 Python 的作用域规则。
    #    这也正是 D2L 的 sgd 为什么要写成 sgd(params, ...) —— 传进去，别从外面抓。


# ============================================================
# 9. 训练
# ============================================================
num_epochs = 10
t0 = time.time()

for epoch in range(num_epochs):
    total_loss, correct, total = 0.0, 0.0, 0.0

    for X, y in train_iter:
        y_hat = net(X)                      # 前向
        l = cross_entropy(y_hat, y)
        l.sum().backward()                  # 反向
        updater(X.shape[0])                 # 更新

        with torch.no_grad():               # 统计指标而已，不需要梯度
            total_loss += float(l.sum())
            correct += accuracy(y_hat, y)
            total += y.numel()

    test_acc = evaluate_accuracy(net, test_iter)
    print("epoch %2d | train loss %.4f | train acc %.4f | test acc %.4f"
          % (epoch + 1, total_loss / total, correct / total, test_acc))

print()
print("总耗时 %.1f 秒" % (time.time() - t0))
print("W.shape =", tuple(W.shape), "  <- (784, 10)，每一列对应一个类别")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
         'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

Wd = W.detach()
fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
for j in range(10):
    ax = axes[j // 5][j % 5]
    col = Wd[:, j].reshape(28, 28)
    ax.imshow(col.numpy(), cmap="RdBu_r",
              vmin=-col.abs().max(), vmax=col.abs().max())
    ax.set_title("%s" % names[j], fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
plt.savefig(r"D:\d2l-study\ch03\W_templates.png", dpi=115, bbox_inches="tight")
print("模板图已保存")

# ============================================================
# 自检
# ============================================================
# 1) 第 1 个 epoch：train loss 约 0.78，test acc 约 0.78
# 2) 第 10 个 epoch：train acc 约 0.85，test acc 约 0.82
# 3) 总耗时约 70~90 秒（CPU）
# 4) 看最后一个 epoch：train acc 比 test acc 高。这说明什么？先猜。
