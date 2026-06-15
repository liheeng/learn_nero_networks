import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================
# 1. 数据
# =====================
np.random.seed(0)

N = 200
X1 = np.random.randn(N // 2, 2) + np.array([2, 2])
X2 = np.random.randn(N // 2, 2) + np.array([-2, -2])
X = np.vstack([X1, X2])

labels = np.array([0] * (N // 2) + [1] * (N // 2))

# =====================
# 2. hidden_dim=2 MLP
# =====================
W1 = np.random.randn(2, 2)
b1 = np.random.randn(2)

W2 = np.random.randn(2, 2)
b2 = np.random.randn(2)


def relu(x):
    return np.maximum(0, x)


# =====================
# 3. forward（保存中间状态）
# =====================
def forward_stages(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    return X, Z1, A1, Z2, A2


X0, Z1, A1, Z2, A2 = forward_stages(X)


# =====================
# 4. 动画插值（关键！）
# =====================
def interpolate(A, B, t):
    return (1 - t) * A + t * B


stages = [(X0, Z1), (Z1, A1), (A1, Z2), (Z2, A2)]

stage_titles = ["X → Linear1", "Linear1 → ReLU1", "ReLU1 → Linear2", "Linear2 → ReLU2"]

# =====================
# 5. 画图
# =====================
fig, ax = plt.subplots()


def update(frame):
    ax.clear()

    stage = frame // 20
    t = (frame % 20) / 20

    A, B = stages[stage]
    X_t = interpolate(A, B, t)

    ax.scatter(X_t[:, 0], X_t[:, 1], c=labels, cmap="coolwarm", s=12)

    ax.set_title(stage_titles[stage])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)


ani = FuncAnimation(fig, update, frames=80, interval=80)

plt.show()
