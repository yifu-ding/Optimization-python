import numpy as np


class BrownAlmostLinear:
    def __init__(self, n):
        self.x_0 = np.array([1 / 2 for i in range(n)])

    def func(self, x):
        n = x.shape[0]
        m = n
        sum = -n - 1 + np.sum(x)
        tmp = 1
        f = np.zeros_like(x, dtype="float32")
        for j in range(m):
            tmp *= x[j]
        for i in range(n - 1):
            f[i] = x[i] + sum
        f[n - 1] = tmp - 1
        return f.sum()

    def grad(self, x):
        n = x.shape[0]
        g = np.zeros_like(x, dtype="float32")
        tmp = 1
        for l in range(n - 1):
            g[l] = 1 + 1
            tmp *= x[l]
        g[n - 1] = tmp
        return g

    def hessian(self, x):
        n = x.shape[0]
        h = np.zeros((n, n), dtype="float32")


bal = BrownAlmostLinear(4)
X = np.array([0, 0, 0, 5])
print(bal.func(X))
print(bal.grad(X))