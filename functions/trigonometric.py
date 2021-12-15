# coding=UTF-8
import numpy as np


class Trigomometric:
    def __init__(self, n):
        self.f_minimun = 0
        self.x_0 = np.array([1. / n for i in range(n)]).reshape(-1, 1)
        self.x_star = None
        self.call_f = 0

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        f = 0.
        sum_cos = 0.
        for i in range(n):
            sum_cos += np.cos(x[i])
        for i in range(n):
            f_i = n - sum_cos + i * (1 - np.cos(x[i])) - np.sin(x[i])
            f += f_i**2
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        g = np.zeros_like(x, dtype="float32")
        sum_cos = 0.
        for i in range(n):
            sum_cos += np.cos(x[i])
        for j in range(n):
            f_j = n - sum_cos + j * (1 - np.cos(x[j])) - np.sin(x[j])
            for i in range(n):
                g[i] += 2 * f_j * (np.sin(x[i]) + i * np.sin(x[i]) -
                                   np.cos(x[i]))
        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = Trigomometric(5)
    print((tri.x_0))
    print(tri.func(tri.x_0))
    print(tri.grad(tri.x_0))