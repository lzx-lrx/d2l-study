# D2L 3.4.3 交叉熵损失：分类问题该用什么损失
# 目标：手写交叉熵，理解它和 softmax 的配合

import torch


# ============================================================
# 一、手写交叉熵（分类版）
# ============================================================
def my_cross_entropy(o, y):
    """TODO(你): 手写交叉熵

       公式： l = -log( 正确类别的预测概率 )

       三步：
         1) 把分数变成概率： p = torch.softmax(o, dim=1)
         2) 取对数：        log_p = torch.log(p)
         3) 把每个样本「正确类别」那一项挑出来，取负、求平均

       hint: 挑出正确类别用高级索引（2.1 学过）：
              picked = log_p[range(len(y)), y]      -> 形状 (batch,)
       hint: 最后 return -picked.mean()

       o: (batch, 类别数)  —— 传「分数」，不是概率！
       y: (batch,)         —— 正确类别的「索引」，不是 one-hot
    """
    p = torch.softmax(o, dim=1)
    log_P = torch.log(p)
    picked =log_P[range(len(y)), y]

    return -picked.mean()


# ============================================================
# 二、验证
# ============================================================
o = torch.tensor([[2.0, 1.0, 0.5],
                  [0.1, 3.0, 0.2],
                  [1.0, 1.0, 1.0]])
y = torch.tensor([0, 1, 2])

print("分数 o =")
print(o)
print("真实类别 y =", y.tolist())
print()

p = torch.softmax(o, dim=1)
print("各类别概率：")
for i in range(3):
    print("   样本 %d: %s   正确类别=%d  它的概率=%.4f  -log=%.4f"
          % (i, [round(v, 4) for v in p[i].tolist()], y[i],
             p[i, y[i]], -torch.log(p[i, y[i]])))
print()

mine = my_cross_entropy(o, y)
official = torch.nn.CrossEntropyLoss()(o, y)
print("我写的交叉熵          =", None if mine is None else round(mine.item(), 6))
print("官方 CrossEntropyLoss =", round(official.item(), 6))
print("一致吗？", mine is not None and torch.allclose(mine, official))


# ============================================================
# 三、看梯度长什么样
# ============================================================
print()
o2 = o.clone().requires_grad_(True)
result = my_cross_entropy(o2, y)
if result is not None:
    result.backward()
    onehot = torch.zeros_like(o)
    onehot[range(len(y)), y] = 1
    print("o.grad                  =")
    print(o2.grad)
    print("(softmax(o) - onehot)/3 =")
    print((torch.softmax(o, dim=1) - onehot) / 3)
    print("一样吗？",
          torch.allclose(o2.grad, (torch.softmax(o, dim=1) - onehot) / 3, atol=1e-6))


# ============================================================
# 自检
# ============================================================
# 1) 我写的交叉熵应该 = 0.557528，和官方一致
# 2) 三个样本各自的损失：0.4644 / 0.1096 / 1.0986
# 3) 梯度 = (预测 - 真实) / batch_size。为什么有个 /3？
#    提示：想想 .mean() 干了什么，再想想 3.2 的 sgd 里为什么也要除以 batch_size
# 4) 把样本 2 的分数从 [1,1,1] 改成 [0,0,0]，它的损失变成多少？
#    两者一样吗？为什么？
