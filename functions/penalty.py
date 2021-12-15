# coding=UTF-8
import numpy as np


class Penalty:
    def __init__(self, n):
        m = n + 1
        self.a = 10e-5
        if n == 4:
            self.f_minimun = 2.24997e-5
        elif n == 10:
            self.f_minimun = 7.08765e-5
        else:
            self.f_minimun = None
        self.x_0 = np.array([j for j in range(n)]).reshape(-1, 1)
        self.x_star = None
        self.call_f = 0

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        tmp = 0
        f = 0
        for i in range(n):
            f_i = np.sqrt(self.a) * (x[i] - 1)
            f += f_i**2
            tmp += x[i]**2
        tmp -= 1 / 4
        f += tmp**2
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        g = np.zeros_like(x, dtype="float32")
        tmp = 0.
        for i in range(n):
            g[i] = 2 * self.a * x[i] - 1
            tmp += x[i]**2
        tmp -= 1 / 4
        for i in range(n):
            g[i] += 4 * x[i] * tmp
        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = Penalty(5)
    print((tri.x_0))
    print(tri.func(tri.x_0))
    print(tri.grad(tri.x_0))