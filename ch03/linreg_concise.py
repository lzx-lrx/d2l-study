 # D2L 3.3 线性回归的简洁实现
# 目标：把 3.2 手搓的零件，全换成 PyTorch 官方组件

import random
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

random.seed(0)          # 固定随机种子：让结果可复现（上一课学到的）
torch.manual_seed(0)


# ============================================================
# 数据：和 3.2 一样，只是改用官方工具
# ============================================================
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """把 TensorDataset 包成 DataLoader —— 官方版的 data_iter"""
    dataset = data.TensorDataset(*data_arrays)    # * 把 (features, labels) 拆成两个参数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


# ============================================================
# TODO(你) 1/2：三个零件，全部换成官方组件
# ============================================================
#   1) net     —— nn.Sequential(nn.Linear(2, 1))    输入 2 维 -> 输出 1 维
#   2) loss    —— nn.MSELoss()
#   3) trainer —— torch.optim.SGD(net.parameters(), lr=0.03)
net     = nn.Sequential(nn.Linear(2, 1))
loss    = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 初始化参数（和 3.2 一样，用很小的随机值）
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# ============================================================
# TODO(你) 2/2：训练循环 —— 注意顺序和 3.2 不一样！
# ============================================================
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 四行，按这个顺序写：
        l = loss(net(X), y)#   1) 前向： l = loss(net(X), y)      <- 不用 .sum() 了
        trainer.zero_grad()#   2) 清零： trainer.zero_grad()      <- 位置提前了！
        l.backward()#   3) 反向： l.backward()
        trainer.step()#   4) 更新： trainer.step()

    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


# ============================================================
# 检查学到了什么
# ============================================================
w = net[0].weight.data
b = net[0].bias.data
print()
print("w 的估计误差 =", [round(v, 6) for v in (true_w - w.reshape(true_w.shape)).tolist()])
print("b 的估计误差 =", round((true_b - b).item(), 6))


# ============================================================
# 自检
# ============================================================
# 1) loss 大约 0.00024 -> 0.00009 -> 0.00009
#    注意：只有 3.2 的一半！因为 nn.MSELoss() 没有那个 /2
# 2) w 误差约 1e-4，b 误差约 1e-3
# 3) 打印 net[0].weight.shape —— 是 (1, 2)，不是 (2, 1)！想想为什么
#    （提示：nn.Linear(in, out) 的 weight 存成 (out, in)）
# 4) 把 trainer.zero_grad() 删掉，loss 会怎样？（你已经知道答案了）
