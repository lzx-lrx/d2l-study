# D2L 3.7 softmax 回归的简洁实现
# 目标：用官方组件重写 3.6，看看两者是不是一回事

import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms

torch.manual_seed(0)


# ============================================================
# 数据（和 3.6 完全一样）
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
# 1. 模型
# ============================================================
# TODO(你) 1/2：用官方组件搭模型
#   nn.Flatten()        <- 顶掉你手写的 X.reshape((-1, W.shape[0]))
#   nn.Linear(784, 10)  <- 顶掉你手写的 W、b 和 torch.matmul
#
#   ⚠️⚠️ 千万不要在后面加 nn.Softmax()！CrossEntropyLoss 内部会自己做。
#          加了就是 double softmax，loss 会卡在 1.8 降不下去。
#
#   hint: nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


# ============================================================
# 2. 损失
# ============================================================
# TODO(你) 2/2：用官方损失
#   hint: nn.CrossEntropyLoss()
#   它 = 你手写的 softmax + cross_entropy，两步合一
loss = nn.CrossEntropyLoss()


# ============================================================
# 3. 优化器
# ============================================================
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


# ============================================================
# 4. 评估
# ============================================================
def evaluate_accuracy(net, data_iter):
    correct = total = 0.0
    with torch.no_grad():
        for X, y in data_iter:
            correct += float((net(X).argmax(1) == y).sum())
            total += y.numel()
    return correct / total


# ============================================================
# 5. 训练
# ============================================================
num_epochs = 10

for epoch in range(num_epochs):
    total_l, n, correct = 0.0, 0, 0.0

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(net(X), y)          # 前向 + 算损失
        trainer.zero_grad()          # 擦白板
        l.backward()                 # 反向
        trainer.step()               # 更新

        with torch.no_grad():
            total_l += float(l) * len(y)
            n += len(y)
            correct += float((y_hat.argmax(1) == y).sum())

    print("epoch %2d | train loss %.4f | train acc %.4f | test acc %.4f"
          % (epoch + 1, total_l / n, correct / n, evaluate_accuracy(net, test_iter)))


# ============================================================
# 6. 和 3.6 手写版对比
# ============================================================
print()
print("参数量 =", sum(p.numel() for p in net.parameters()), "  <- 手写版也是 7850")
print("net[0] =", net[0])
print("net[1] =", net[1])
print("net[1].weight.shape =", tuple(net[1].weight.shape), "  <- 注意是 (10, 784)")
print("net[1].bias.shape   =", tuple(net[1].bias.shape))
print()
print("对比 3.6：你的 W 是 (784, 10)。两者是转置关系。为什么？（3.3 讲过）")


# ============================================================
# 自检
# ============================================================
# 1) 10 轮之后 test acc 应该到 82%~84%
# 2) 参数量 = 7850，和手写版一模一样
# 3) net[1].weight.shape 是 (10, 784) —— 为什么不是 (784, 10)？
# 4) 扩展：在 nn.Sequential 里加一个 nn.Softmax(dim=1)，
#    跑 3 轮就行，看 loss 卡在多少、test acc 掉到多少
