# D2L 3.5 图像分类数据集 —— Fashion-MNIST
# 目标：加载真实图片，第一次"看见"数据

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")          # 不开窗口，直接存成图片文件
import matplotlib.pyplot as plt


# ============================================================
# 1. 加载数据集
# ============================================================
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root=r"D:\d2l-study\data", train=True,  transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root=r"D:\d2l-study\data", train=False, transform=trans, download=True)

print("训练集:", len(mnist_train), "张")
print("测试集:", len(mnist_test), "张")

x, y = mnist_train[0]
print()
print("单张图 shape =", tuple(x.shape), "   <- (通道, 高, 宽)")
print("单张图 dtype =", x.dtype)
print("像素范围     = [%.3f, %.3f]" % (x.min(), x.max()))
print("标签         =", y)


# ============================================================
# 2. 十个类别
# ============================================================
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def get_labels(labels):
    return [text_labels[int(i)] for i in labels]


print()
for i, name in enumerate(text_labels):
    print("   %d : %s" % (i, name))


# ============================================================
# 3. DataLoader 取一个批次
# ============================================================
batch_size = 18
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)

X, y = next(iter(train_iter))
print()
print("一个 batch: X.shape =", tuple(X.shape), "  y.shape =", tuple(y.shape))
print("           y =", y.tolist())
print("           对应的名字 =", get_labels(y))


# ============================================================
# 4. 画出来看看
# ============================================================
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.6):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy(), cmap="gray")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=8)
    return axes


# TODO(你) 1/2: 把 X 从 (18, 1, 28, 28) 变成 (18, 28, 28)，好画图
#   hint: X.reshape(18, 28, 28)
imgs = X.reshape(18, 28, 28)

if imgs is not None:
    show_images(imgs, 2, 9, titles=get_labels(y))
    out = r"D:\d2l-study\ch03\fashion_mnist_sample.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print()
    print("图片已保存到:", out)


# ============================================================
# 5. 为 3.6 做准备：拉平
# ============================================================
# TODO(你) 2/2: 3.6 要用 nn.Linear(784, 10)，输入必须是 784 维。
#               把 X 从 (18, 1, 28, 28) 变成 (18, 784)
#   hint: X.reshape(X.shape[0], -1)     <- -1 让 PyTorch 自己算
X_flat = X.reshape(X.shape[0], -1)
print()
print("X_flat.shape =", None if X_flat is None else tuple(X_flat.shape))


# ============================================================
# 自检
# ============================================================
# 1) 训练集 60000 张，测试集 10000 张
# 2) 单张图 shape = (1, 28, 28) —— 注意通道在最前面
# 3) 打开 fashion_mnist_sample.png 看一眼，回答：
#      - 哪几个类别容易混？（提示：看名字）
#      - 如果模型完全瞎猜，10 个类别，精度大约多少？
# 4) X_flat.shape 应该是 (18, 784)。784 = 28 x 28 x 1
