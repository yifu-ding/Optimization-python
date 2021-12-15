# coding=UTF-8
import numpy as np


class ExtendedRosenbrock:
    def __init__(self, n):
        x_0 = []
        for i in range(n // 2):
            x_0.append(-1.2)
            x_0.append(1.)
        self.x_0 = np.array(x_0, dtype="float32").reshape(-1, 1)
        self.f_minimun = 0
        self.x_star = np.array([1 for i in range(n)]).reshape(-1, 1)
        self.call_f = 0

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        f = 0.
        for i in range(1, (n // 2) + 1):
            f_1 = 10. * (x[2 * i] - x[i - 1]**2)
            f_2 = 1. - x[2 * i - 1]
            f = f + f_1**2 + f_2**2
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        g = np.zeros(n + 1, dtype="float32")
        for i in range(1, (n // 2) + 1):
            f_1 = 10. * (x[2 * i] - x[i - 1]**2)
            f_2 = 1. - x[2 * i - 1]
            g[2 * i] += 2 * f_1 * 10
            g[i - 1] += 2 * f_1 * (-20 * x[i - 1])
            g[2 * i - 1] += 2 * f_2 * (-1)
        g = np.delete(g, 0)
        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = ExtendedRosenbrock(5)
    # print((tri.x_star))
    # print(tri.func(tri.x_star))
    print(tri.grad(tri.x_star))