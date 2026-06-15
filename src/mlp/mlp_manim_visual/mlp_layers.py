import numpy as np


def relu(x):
    return np.maximum(0, x)


class SimpleMLP:
    def __init__(self):
        self.W1 = np.array([[1.2, 0.7], [-0.6, 1.0]])
        self.b1 = np.array([0.5, -0.3])

        self.W2 = np.array([[1.0, -0.8], [0.9, 1.1]])
        self.b2 = np.array([0.1, 0.2])

    def forward_stage(self, X):
        stages = []

        # input
        stages.append(X.copy())

        # layer 1
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        stages.append(A1)

        # layer 2
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        stages.append(A2)

        return stages
