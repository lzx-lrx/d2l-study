# D2L 3.4.2 softmax：把"分数"变成"概率"
# 目标：理解分类问题的输出为什么是一组概率，以及 softmax 干了什么

import torch


# ============================================================
# 一、手动实现 softmax（核心）
# ============================================================
def my_softmax(o):
    """TODO(你): 实现 softmax 公式

       ŷ_j = exp(o_j) / sum_k exp(o_k)

       hint: 两行 —— 先 e = torch.exp(o)，再 return e / e.sum(dim=-1, keepdim=True)
       hint: 用 dim=-1 是为了同时兼容一维 (3,) 和二维 (1, 3)
    """
    e = torch.exp(o)
    return e / e.sum(dim = -1, keepdim=True)

o = torch.tensor([1.0, 2.0, 3.0])
print("分数 o           =", o.tolist())
print("我写的 softmax   =", my_softmax(o))
print("官方的 softmax   =", torch.softmax(o, dim=0))
print("我的加起来 =", my_softmax(o).sum().item())
print("和官方一致吗？", torch.allclose(my_softmax(o), torch.softmax(o, dim=0)))


# ============================================================
# 二、看 exp 怎么"放大差距"
# ============================================================
print()
print("同样的分数差，经过 softmax 后差多少：")
for lst in ([1., 1., 1.], [1., 2., 3.], [1., 5., 9.], [1., 10., 20.]):
    o = torch.tensor(lst)
    p = my_softmax(o)
    print("  o = %-18s ->  概率 = %s" % (str([int(v) for v in lst]),
          [round(v, 4) for v in p.tolist()]))


# ============================================================
# 三、分类场景：三个类别
# ============================================================
print()
classes = ["猫", "鸡", "狗"]

o = torch.tensor([[.0, .0, .0]])       # 模型对一张图算出的分数
print("这张图的分数 o =", o.tolist())
probs = my_softmax(o)
print("各类别概率：")
for name, p in zip(classes, probs[0].tolist()):
    print("    %-2s : %.4f  |%s" % (name, p, "#" * int(p * 40)))
print("预测类别 =", classes[probs.argmax().item()])
print("one-hot 形式的真实标签（假设真的是猫）= [1, 0, 0]")


# ============================================================
# 自检
# ============================================================
# 1) my_softmax([1,2,3]) 应该 = [0.0900, 0.2447, 0.6652]，和 torch.softmax 一致
# 2) o=[1,1,1] 那行应该输出 [0.3333, 0.3333, 0.3333]
# 3) 这张图的概率应该约 [0.6285, 0.2312, 0.1403]，预测"猫"
# 4) 把 o 改成 [[0., 0., 0.]]，输出是什么？为什么？
# 5) 想一想：softmax 之后，三个数的"大小顺序"会变吗？
