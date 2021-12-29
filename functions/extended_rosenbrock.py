# coding=UTF-8
import numpy as np
# 21


class ExtendedRosenbrock:
    def __init__(self, m):
        n = self.m = self.n = m
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
        fi = np.zeros(self.m, dtype="float32")
        fi[0] = 10 * x[1]
        fi[1] = 1 - x[0]
        for i in range(2, (n // 2) + 1):
            fi[2 * i - 2] = 10 * (x[2 * i - 1] - x[i - 2]**2)
            fi[2 * i - 1] = 1 - x[2 * i - 2]
        f = np.sum(fi.T @ fi)
        return f

    def grad(self, x):
        self.call_f += 1
        n = x.shape[0]
        fi = np.zeros(self.m, dtype="float32")
        gi = np.zeros((self.n, self.m), dtype="float32")
        g = np.zeros(self.n, dtype="float32").reshape(-1, 1)

        fi[0] = 10 * x[1]
        fi[1] = 1 - x[0]
        gi[1][0] = 10
        gi[0][1] = -1
        for i in range(2, (n // 2) + 1):
            fi[2 * i - 2] = 10 * (x[2 * i - 1] - x[i - 2]**2)
            fi[2 * i - 1] = 1 - x[2 * i - 2]
            gi[2 * i - 1][2 * i - 2] = 10
            gi[i - 2][2 * i - 2] = -20 * x[i - 2]
            gi[2 * i - 2][2 * i - 1] = -1

        for i in range(n):
            g[i] = np.sum(2 * gi[i].T @ fi)

        return g

    def hessian(self, x):
        return


if __name__ == '__main__':
    tri = ExtendedRosenbrock(4)
    print((tri.x_0))
    print(tri.func(tri.x_0))
    tri.grad(tri.x_0)