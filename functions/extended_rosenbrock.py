# coding=UTF-8
import numpy as np
# 21


class ExtendedRosenbrock:
    def __init__(self, m):
        n = m
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
            f_1 = 10. * (x[2 * i - 1] - x[i - 2]**2)
            f_2 = 1. - x[2 * i - 2]
            f = f + f_1**2 + f_2**2
        # r = np.zeros(n)
        # for i in range(1, n // 2 + 1):
        #     r[i] = 1.0 - x[2 * i - 2]
        #     r[i + 1] = 10.0 * (x[2 * i - 1] - x[i - 1]**2)
        #     f = f + r[i]**2 + r[i + 1]**2
        # f = np.dot(r, r)
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        g = np.zeros(n, dtype="float32")
        r = np.zeros(n)
        for i in range(1, n // 2 + 1):
            r[i] = 1.0 - x[2 * i - 2]
            r[i + 1] = 10.0 * (x[2 * i - 1] - x[i - 1]**2)
            # f = f + r[i]**2 + r[i + 1]**2
        # print(r)
        for i in range(1, n // 2 + 1):
            g[2 * i - 2] += -1 * 2 * r[i]
            g[2 * i - 1] += 10 * 2 * r[i + 1]
            g[i - 1] += 10 * (-2) * (x[i - 1]) * r[i + 1]
        # for i in range(1, (n // 2) + 1):
        #     f_1 = 10. * (x[2 * i - 1] - x[i - 2]**2)
        #     f_2 = 1. - x[2 * i - 2]
        #     g[2 * i - 1] += 2 * f_1 * 10
        #     g[i - 2] += 2 * f_1 * (-20 * x[i - 2])
        #     g[2 * i - 2] += 2 * f_2 * (-1)
        # g = np.delete(g, 0)
        # print(g)
        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = ExtendedRosenbrock(4)
    print((tri.x_0))
    # print(tri.func(tri.x_star))
    tri.grad(tri.x_0)