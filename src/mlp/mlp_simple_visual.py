import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(0)

# ===== 1. 造数据（2D点云）=====
N = 300

# 两个高斯簇
X1 = np.random.randn(N // 2, 2) + np.array([2, 2])
X2 = np.random.randn(N // 2, 2) + np.array([-2, -2])

X = np.vstack([X1, X2])

# ===== 2. 定义MLP参数 =====
W1 = np.random.randn(2, 2)
b1 = np.random.randn(2)

W2 = np.random.randn(2, 2)
b2 = np.random.randn(2)


def relu(x):
    return np.maximum(0, x)


# ===== 3. 各层输出 =====
X0 = X
X1_out = relu(X0 @ W1 + b1)
X2_out = relu(X1_out @ W2 + b2)

stages = [X0, X1_out, X2_out]

titles = [
    "Input Space",
    "After Linear + ReLU (Layer 1)",
    "After Linear + ReLU (Layer 2)",
]

# ===== 4. 动画 =====
fig, ax = plt.subplots()


def update(i):
    ax.clear()
    X_ = stages[i]

    ax.scatter(
        X_[:, 0],
        X_[:, 1],
        c=np.concatenate([np.zeros(N // 2), np.ones(N // 2)]),
        cmap="coolwarm",
        s=10,
    )

    ax.set_title(titles[i])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


ani = FuncAnimation(fig, update, frames=3, interval=1200)

plt.show()
