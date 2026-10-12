# D2L 3.1.1.4 随机梯度下降：手动走一步
# 目标：亲手用梯度把 w 和 b 改好一点点

import torch

# ---------- 数据 ----------
X = torch.tensor([[100.,   5.],
                  [ 80.,  10.],
                  [120.,   2.]])
y = torch.tensor([320., 230., 420.])

# ---------- 参数：这次开启梯度追踪 ----------
# requires_grad=True —— 和 ch02/autograd.py 里一样
w = torch.tensor([2.5, -6.0], requires_grad=True)
b = torch.tensor(140.0,      requires_grad=True)


def loss_fn():
    """零件 1 + 零件 2 合起来：从参数算出损失"""
    y_hat = torch.matmul(X, w) + b
    return torch.mean(0.5 * (y_hat - y)**2)


# ---------- 第 0 步：初始状态 ----------
L0 = loss_fn()
print("初始 L  =", L0.item())

L0.backward()                      # 零件 3 的前半段：自动算梯度
print("dL/dw  =", w.grad)
print("dL/db  =", b.grad)


# ---------- 第 1 步：走一步梯度下降 ----------
lr = 0.0001                        # 学习率

# TODO(你): 用 w.grad 和 b.grad 更新 w 和 b
#   hint: 梯度是正数 -> w 变大 L 就变大 -> 所以要减
#   hint: 两行，形如 `w -= lr * w.grad`，写在 with torch.no_grad(): 里面
with torch.no_grad():
 w -= lr * w.grad
 b -= lr * b.grad

print("更新后 w =", w)
print("更新后 b =", b)


# ---------- 第 2 步：损失降了吗 ----------
w.grad = None                      # 清掉旧梯度，否则下次 backward 会累加
b.grad = None

L1 = loss_fn()
print("新的 L  =", L1.item())
print("下降了  =", L0.item() - L1.item())


# ---------- 自检 ----------
# 1) 新的 L 应该约等于 257.85（从 694 降下来）
# 2) 3 套房子是不是都变准了？如果有一套反而变差了，想想为什么整体还是变好
# 3) 扩展：把 lr 改成 0.0005，会更好还是更糟？先猜再跑
