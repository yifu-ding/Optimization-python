import numpy as np


class EPS:
    def __init__(self):
        pass

    def func(self, x):
        N = x.shape[0]
        assert N % 4 == 0
        f = 0
        for i in range(N // 4):
            r_1 = x[4*i] + 10 * x[4*i+1]
            r_2 = np.power(5, 0.5) * (x[4*i+2] - x[4*i+3])
            r_3 = np.power(x[4*i+1] - 2 * x[4*i+2], 2)
            r_4 = np.power(10, 0.5) * np.power(x[4*i] - x[4*i+3], 2)
            f = r_1 * r_1 + r_2 * r_2 + r_3 * r_3 + r_4 * r_4 + f
        return f
    
    def grad(self, x):
        N = x.shape[0]
        assert N % 4 == 0
        g = np.zeros((N,), dtype=np.float64)
        for i in range(N // 4):
            r_1 = x[4*i] + 10 * x[4*i+1]
            r_2 = np.power(5, 0.5) * (x[4*i+2] - x[4*i+3])
            r_3 = np.power(x[4*i+1] - 2 * x[4*i+2], 2)
            r_4 = np.power(10, 0.5) * np.power(x[4*i] - x[4*i+3], 2)
            gr = np.zeros_like(g)
            gr[4*i] = 1
            gr[4*i+1] = 10
            g = g + 2 * r_1 * gr
            gr = np.zeros_like(g)
            gr[4*i+2] = np.power(5, 0.5)
            gr[4*i+3] = -np.power(5, 0.5)
            g = g + 2 * r_2 * gr
            gr = np.zeros_like(g)
            gr[4*i+1] = 2 * (x[4*i+1] - 2*x[4*i+2])
            gr[4*i+2] = 2 * -2 * (x[4*i+1] - 2*x[4*i+2])
            g = g + 2 * r_3 * gr
            gr = np.zeros_like(g)
            gr[4*i] = np.power(10, 0.5) * 2 * (x[4*i] - x[4*i+3])
            gr[4*i+3] = -np.power(10, 0.5) * 2 * (x[4*i] - x[4*i+3])
            g = g + 2 * r_4 * gr

        return g


if __name__ == '__main__':
    eps = EPS()
    X = np.array([1, 2, 3, 4])
    print(eps.func(X))
    print(eps.grad(X))
