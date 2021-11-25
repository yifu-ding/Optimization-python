# coding=UTF-8
import numpy as np


class BrownAndDennis:
    def __init__(self, m=20):
        self.n = 4
        self.m = m
        assert self.m >= self.n
        self.x_0 = np.array([25, 5, -5, -1], dtype="float32").reshape(-1, 1)
        self.f_minimun = 85822.2
        self.call_f = 0
        self.x_star = None

    def func(self, x):
        self.call_f += 1
        f = 0
        for l in range(self.m):
            t = (l+1) / 5.
            f_l = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) - np.cos(t))**2
            f = f + (f_l)**2
        return f

    def grad(self, x):
        g = np.zeros_like(x, dtype="float32")
        for l in range(self.m):
            t = (l+1) / 5.
            f_l = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) - np.cos(t))**2
            g[0] += 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t))
            g[1] += 2 * f_l * 2 * (x[0] + t * x[1] - np.exp(t)) * t
            g[2] += 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) - np.cos(t))
            g[3] += 2 * f_l * 2 * (x[2] + x[3] * np.sin(t) - np.cos(t)) * np.sin(t)
        return g

if __name__ == '__main__':
    bad = BrownAndDennis(m=20)
    x = bad.x_0
    # x = np.array([1, 2, 3, 4])
    print(bad.func(x))
    print(bad.grad(x))