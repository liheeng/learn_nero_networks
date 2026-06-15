import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 1. 数据
# =========================
np.random.seed(0)

N = 200
X1 = np.random.randn(N//2, 2) + np.array([2, 2])
X2 = np.random.randn(N//2, 2) + np.array([-2, -2])

X = np.vstack([X1, X2])
labels = np.array([0]*(N//2) + [1]*(N//2))

# =========================
# 2. Neural ODE vector field f(x)
# =========================
W1 = np.random.randn(2, 16) * 0.5
b1 = np.random.randn(16) * 0.1
W2 = np.random.randn(16, 2) * 0.5
b2 = np.random.randn(2) * 0.1

def relu(x):
    return np.maximum(0, x)

def f(x):
    h = relu(x @ W1 + b1)
    return h @ W2 + b2

# =========================
# 3. ODE integration (Euler)
# =========================
def ode_step(x, dt=0.05):
    return x + dt * f(x)

def rollout(X, steps=40):
    traj = [X.copy()]
    x = X.copy()

    for _ in range(steps):
        x = ode_step(x)
        traj.append(x.copy())

    return traj

traj = rollout(X)

# =========================
# 4. decision boundary grid
# =========================
grid_size = 60
x_lin = np.linspace(-6, 6, grid_size)
y_lin = np.linspace(-6, 6, grid_size)
xx, yy = np.meshgrid(x_lin, y_lin)

grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

def classify(x):
    xT = x.copy()
    for _ in range(40):
        xT = ode_step(xT)
    return xT

# =========================
# 5. animation
# =========================
fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    # ===== trajectory =====
    X_t = traj[frame]

    ax.scatter(
        X_t[:, 0], X_t[:, 1],
        c=labels,
        cmap="coolwarm",
        s=15
    )

    # ===== decision boundary =====
    grid_t = classify(grid)

    # simple linear classifier in final space
    score = grid_t[:, 0] + grid_t[:, 1]
    score = score.reshape(xx.shape)

    ax.contourf(xx, yy, score, levels=20, cmap="coolwarm", alpha=0.3)

    ax.set_title(f"Neural ODE Flow t={frame}")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

ani = FuncAnimation(fig, update, frames=len(traj), interval=100)

plt.show()