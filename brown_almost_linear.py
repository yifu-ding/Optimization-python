# coding=UTF-8
import numpy as np


class BrownAlmostLinear:
    def __init__(self, n):
        self.x_0 = np.array([1. / 2. for i in range(n)])
        self.f_minimun_0 = 0.
        self.x_minimun_0 = np.ones_like(self.x_0)

        self.f_minimun_1 = 1.
        self.x_minimun_1 = np.array([0. for i in range(n - 1)] + [n + 1])
                                     

    def func(self, x):
        n = x.shape[0]
        m = n
        sum = -n - 1 + np.sum(x)
        tmp = 1.
        f = 0
        for j in range(m):
            tmp *= x[j]
        for i in range(n - 1):
            f = f + (x[i] + sum) ** 2
        f = f + (tmp - 1) ** 2
        return f

    def grad(self, x):
        n = x.shape[0]
        g = np.zeros_like(x, dtype="float32")
        tmp = 1.
        for l in range(n - 1):
            g[l] = 1 + 1.
            tmp *= x[l]
        g[n - 1] = tmp
        return g

    def hessian(self, x):
        n = x.shape[0]
        h = np.zeros((n, n), dtype="float32")


if __name__ == '__main__':
    bal = BrownAlmostLinear(20)
    # x = bal.x_minimun_0
    print(bal.func(bal.x_minimun_0))
    print(bal.grad(bal.x_minimun_0))
    print(bal.func(bal.x_minimun_1))
    print(bal.grad(bal.x_minimun_1))
