# coding=UTF-8
import numpy as np


class ExtendedPowellSingular:
    def __init__(self, m=None):
        self.x_0 = np.array([3, -1, 0, 1], dtype="float32").reshape(-1, 1)
        self.f_minimun = None
        self.call_f = 0
        self.x_star = None
        pass

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        assert n % 4 == 0
        m = n
        f = 0
        for i in range(m // 4):
            f_1 = x[4 * i] + 10 * x[4 * i + 1]
            f_2 = np.power(5, 0.5) * (x[4 * i + 2] - x[4 * i + 3])
            f_3 = np.power(x[4 * i + 1] - 2 * x[4 * i + 2], 2)
            f_4 = np.power(10, 0.5) * np.power(x[4 * i] - x[4 * i + 3], 2)
            f = f_1 * f_1 + f_2 * f_2 + f_3 * f_3 + f_4 * f_4 + f
        return f

    def grad(self, x):
        n = x.shape[0]
        # import pdb
        # pdb.set_trace()
        assert n % 4 == 0
        m = n
        g = np.zeros((n, ), dtype=np.float64)
        for i in range(m // 4):
            f_1 = x[4 * i] + 10 * x[4 * i + 1]
            f_2 = np.power(5, 0.5) * (x[4 * i + 2] - x[4 * i + 3])
            f_3 = np.power(x[4 * i + 1] - 2 * x[4 * i + 2], 2)
            f_4 = np.power(10, 0.5) * np.power(x[4 * i] - x[4 * i + 3], 2)
            gf = np.zeros_like(g)
            gf[4 * i] = 1
            gf[4 * i + 1] = 10
            g = g + 2 * f_1 * gf
            gf = np.zeros_like(g)
            gf[4 * i + 2] = np.power(5, 0.5)
            gf[4 * i + 3] = -np.power(5, 0.5)
            g = g + 2 * f_2 * gf
            gf = np.zeros_like(g)
            gf[4 * i + 1] = 2 * (x[4 * i + 1] - 2 * x[4 * i + 2])
            gf[4 * i + 2] = 2 * -2 * (x[4 * i + 1] - 2 * x[4 * i + 2])
            g = g + 2 * f_3 * gf
            gf = np.zeros_like(g)
            gf[4 * i] = np.power(10, 0.5) * 2 * (x[4 * i] - x[4 * i + 3])
            gf[4 * i + 3] = -np.power(10, 0.5) * 2 * (x[4 * i] - x[4 * i + 3])
            g = g + 2 * f_4 * gf

        return g


if __name__ == '__main__':
    eps = ExtendedPowellSingular()
    X = np.array([0, 0, 0, 0])
    print(eps.func(X))
    print(eps.grad(X))
