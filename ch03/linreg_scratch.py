# D2L 3.2 线性回归的从零开始实现
# 目标：把 3.1 的三个零件组装成真正的训练循环

import random
import torch

random.seed(0)          # TODO(你): 加这两行
torch.manual_seed(0)
# ============================================================
# 3.2.1 生成合成数据集
# ============================================================
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))   # 1000 x 2，标准正态
    y = torch.matmul(X, w) + b                       # 用「真参数」算出真值
    y += torch.normal(0, 0.01, y.shape)              # 加一点点噪声
    return X, y.reshape((-1, 1))


# 这两行是「标准答案」——只有我们自己知道
true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)
print("features.shape =", features.shape)      # 1000 x 2
print("labels.shape   =", labels.shape)        # 1000 x 1
print("features[:3]   =", features[:3].tolist())
print("labels[:3]     =", labels[:3].flatten().tolist())
print()


# ============================================================
# 3.2.2 读取数据集（2.x 讲过，直接给）
# ============================================================
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)                      # 每个 epoch 只洗一次
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10


# ============================================================
# 3.2.3 初始化模型参数
# ============================================================
# 随机小值初始化。形状必须是 (2, 1)，好和 labels 的 (batch, 1) 对上
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# ============================================================
# 3.2.4 定义模型
# ============================================================
def linreg(X, w, b):
    """线性回归模型 —— 零件 1"""
    return torch.matmul(X, w) + b


# ============================================================
# 3.2.5 定义损失函数
# ============================================================
def squared_loss(y_hat, y):
    """平方损失。注意：只除以 2，没有除以 n"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ============================================================
# 3.2.6 定义优化算法
# ============================================================
def sgd(params, lr, batch_size):
    """小批量随机梯度下降 —— 零件 3
       注意 / batch_size 在这里，所以 squared_loss 里不用再取平均"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 起个别名，跟书里写法一致
net = linreg
loss = squared_loss


# ============================================================
# 3.2.7 训练
# ============================================================
lr = 0.03
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # TODO(你): 按顺序写三行 —— 这是全书所有模型的训练循环骨架
        l = loss(net(X, w, b), y) #   1) 前向 —— 用 loss(net(X, w, b), y) 算出本批次损失，存进变量 l
        l.sum().backward()#   2) 反向 —— 对 l 求和后调用 backward()
        sgd([w, b], lr, batch_size)#   3) 更新 —— 调用 sgd([w, b], lr, batch_size)
        #   hint: 对照 gd_step.py 的第 0/1 步，只是这里包成了函数、还多了 batch 维度

    # 每个 epoch 结束，在「全量数据」上评估一次
with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# ============================================================
# 3.2.8 学到什么了？
# ============================================================
print()
print("真实 w   =", true_w.tolist(), "   真实 b   =", true_b)
print("学到的 w =", [round(v, 4) for v in w.reshape(true_w.shape).tolist()],
      "   学到的 b =", round(b.item(), 4))
print("w 的误差 =", [round(v, 5) for v in (true_w - w.reshape(true_w.shape)).tolist()])
print("b 的误差 =", round((true_b - b).item(), 5))


# ============================================================
# 自检
# ============================================================
# 1) 三个 epoch 的 loss 大约 0.03 -> 0.0001 -> 0.00005（每次跑略有不同）
# 2) 学到的 w 应极接近 [2, -3.4]，b 极接近 4.2（误差在 0.001 量级）
# 3) 把 num_epochs 改成 0，w 和 b 会是什么？（先猜再跑）
# 4) 把 lr 改成 2 会怎样？改成 0.5 呢？（先猜再跑）
#    提示：3.1 里 lr=0.0005 就炸了，为什么这里 0.5 反而没事？
