# coding=UTF-8
import numpy as np
# 23


class Penalty:
    def __init__(self, m):
        self.n = m
        self.m = self.n + 1
        self.a = 1e-5
        if self.n == 4:
            self.f_minimun = 2.24997e-5
        elif self.n == 10:
            self.f_minimun = 7.08765e-5
        else:
            self.f_minimun = None
        # self.x_0 = np.array([1 for j in range(self.n)]).reshape(-1, 1)
        self.x_0 = np.random.rand(self.n).reshape(-1, 1)
        self.x_star = None
        self.call_f = 0

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        fi = np.zeros(self.m, dtype="float32")
        for i in range(n):
            fi[i] = np.sqrt(self.a) * (x[i] - 1)
        fi[n] = np.sum(x.T @ x) - 0.25
        f = np.sum(fi.T @ fi)
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        gi = np.zeros((n, self.m), dtype="float32")
        g = np.zeros_like(x, dtype="float32")
        for i in range(n):
            gi[i][i] = np.sqrt(self.a)
            gi[i][self.m - 1] = (2 * x)[i]

        fi = np.zeros(self.m, dtype="float32")
        for i in range(n):
            fi[i] = np.sqrt(self.a) * (x[i] - 1)
        fi[n] = np.sum(x.T @ x) - 0.25

        for i in range(n):
            g[i] = np.sum(2 * gi[i].T @ fi)

        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = Penalty(5)
    print((tri.x_0))
    print(tri.func(tri.x_0))
    print(tri.grad(tri.x_0))