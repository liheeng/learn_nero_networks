import numpy as np


def make_two_clusters(n=200):
    np.random.seed(42)

    c1 = np.random.randn(n // 2, 2) + np.array([2, 2])
    c2 = np.random.randn(n // 2, 2) + np.array([-2, -2])

    X = np.vstack([c1, c2])

    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    return X, labels
