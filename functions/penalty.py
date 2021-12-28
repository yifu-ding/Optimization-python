# coding=UTF-8
import numpy as np
# 14


class Penalty:
    def __init__(self, m):
        n = m
        m = n + 1
        self.a = 10e-5
        if n == 4:
            self.f_minimun = 2.24997e-5
        elif n == 10:
            self.f_minimun = 7.08765e-5
        else:
            self.f_minimun = None
        self.x_0 = np.array([j + 1 for j in range(n)]).reshape(-1, 1)
        # print("self.x_0=" + str(self.x_0))
        self.x_star = None
        self.call_f = 0

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        tmp = 0.
        f = 0.
        for i in range(n):
            f_i = np.sqrt(self.a) * (x[i] - 1)
            f += f_i**2
            tmp += x[i]**2
        tmp -= 0.25
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
        tmp -= 0.25
        for i in range(n):
            g[i] += 4 * x[i] * tmp
        return g

    def hessian(self, x):
        n = len(x)
        a = 1e-5
        # x = x(:);
        # r = np.sqrt(a) * (x - 1).append(x.T * x - 0.25)
        # import pdb
        # pdb.set_trace()
        r = np.sqrt(a) * (x - 1)
        r = np.append(r, x.T @ x - 0.25)
        # J = np.sqrt(a) * np.eye(n).append(2 * x.T)
        J = np.sqrt(a) * np.eye(n)
        J = np.append(J, 2 * x.T)
        G = 2 * (J.T @ J + 2 * np.eye(n) * r[n])
        return G


if __name__ == '__main__':
    tri = Penalty(5)
    print((tri.x_0))
    print(tri.func(tri.x_0))
    print(tri.grad(tri.x_0))