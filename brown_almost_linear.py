# coding=UTF-8
import numpy as np


class BrownAlmostLinear:
    def __init__(self, n):
        self.x_0 = np.array([1. / 2. for i in range(n)])

        self.f_minimun_0 = 0.
        self.x_minimun_0 = np.ones_like(x_0)

        self.f_minimun_1 = 1.
        self.x_minimun_1 = np.append(np.array([0. for i in range(n - 1)]),
                                     n + 1,
                                     dtype="float32")

    def func(self, x):
        n = x.shape[0]
        m = n
        sum = -n - 1 + np.sum(x)
        tmp = 1.
        for j in range(m):
            tmp *= x[j]
        for i in range(n - 1):
            f[i] = x[i] + sum
        f[n - 1] = tmp - 1
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