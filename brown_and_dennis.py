# coding=UTF-8
import numpy as np


class BrownAndDennis:
    def __init__(self, m=20):
        self.n = 4
        self.m = m
        assert self.m >= self.n
        self.x_0 = np.array([25, 5, -5, -1], dtype="float32").reshape(-1, 1)
        if m == 20:
            self.f_minimun = 85822.2
        else:
            self.f_minimun = None
        self.call_f = 0
        self.x_star = None

    def func(self, x):
        self.call_f += 1
        f = 0
        for l in range(self.m):
            t = (l + 1) / 5.
            f_l = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) -
                                                      np.cos(t))**2
            f = f + (f_l)**2
        return f

    def grad(self, x):
        self.call_f += 1
        g = np.zeros_like(x, dtype="float32")
        for l in range(self.m):
            t = (l + 1) / 5.
            f_l = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) -
                                                      np.cos(t))**2
            g[0] += 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t))
            g[1] += 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t)) * t
            g[2] += 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) - np.cos(t))
            g[3] += 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) -
                                   np.cos(t)) * np.sin(t)
        return g

    def hessian(self, x):
        self.call_f += 1
        h = np.zeros((self.n, self.n))
        for l in range(self.m):
            t = (l + 1) / 5.
            f_l = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) -
                                                      np.cos(t))**2
            g_0 = 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t))
            h[0][0] += 2 * f_l * 2
            h[0][1] += 2 * f_l * 2 * t
            g_1 = 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t)) * t
            h[1][0] += 2 * f_l * 2 * t
            h[1][1] += 2 * f_l * 2 * t * t
            g_2 = 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) - np.cos(t))
            h[2][2] += 2 * f_l * 2
            h[2][3] += 2 * f_l * 2 * np.sin(t)
            g_3 = 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) -
                                 np.cos(t)) * np.sin(t)
            h[3][2] += 2 * f_l * 2 * np.sin(t)
            h[3][3] += 2 * f_l * 2 * np.sin(t) * np.sin(t)
        return h


if __name__ == '__main__':
    bad = BrownAndDennis(m=20)
    x = bad.x_0
    # x = np.array([1, 2, 3, 4])
    print(bad.func(x))
    print(bad.grad(x))