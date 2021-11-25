# coding=UTF-8
import numpy as np


class BrownAlmostLinear:
    def __init__(self, n):
        self.x_0 = np.append(np.array([0. for i in range(n - 1)]),
                             10).reshape(-1, 1)
        self.f_minimun_0 = 0.
        self.x_minimun_0 = np.ones_like(self.x_0)

        self.f_minimun = 1.
        self.x_star = np.array([0. for i in range(n - 1)] + [n + 1])
        self.call_f = 0
        

    def func(self, x):
        self.call_f += 1
        n = x.shape[0]
        m = n
        sum = -n - 1 + np.sum(x)
        tmp = 1.
        f = 0
        for j in range(m):
            tmp *= x[j]
        for i in range(n - 1):
            f = f + (x[i] + sum)**2
        f = f + (tmp - 1)**2
        return f

    def grad(self, x):
        n = x.shape[0]
        g = np.zeros_like(x, dtype="float32")
        sum = -n - 1 + np.sum(x)
        tmp = 1.
        for j in range(n):
            tmp *= x[j]
        f_n = tmp - 1
        for l in range(n - 1):
            f = x[l] + sum
            g = g + 2 * f
            g[l] = g[l] + 2 * f
            tmp = 1.
            for k in range(n):
                if k == l:
                    continue
                tmp = tmp * x[k]
            g[l] = g[l] + 2 * f_n * tmp

        tmp = 1.
        for k in range(n):
            if k == n - 1:
                continue
            tmp = tmp * x[k]
        g[n - 1] = g[n - 1] + 2 * f_n * tmp
        return g

    def hessian(self, x):
        n = x.shape[0]
        J = np.zeros((n, n))
        sum = np.sum(x)
        for l in range(n - 1):
            J[l, :] = 1
            J[l, l] = 2
        
        def mul_not_k(x, k1, k2):
            tmp = 1.
            for kk in range(n):
                if kk != k1 and kk != k2:
                    tmp = tmp * x[kk]
            return tmp
        for j in range(n):
            J[n-1, j] = mul_not_k(x, j, j)
        # print(J)

        G = np.dot(J.T, J)

        F__ = np.zeros_like(G)
        for i in range(n):
            for j in range(n):
                F__[i, j] = mul_not_k(x, i, j)
            F__[i, i] = 0
        f_n = mul_not_k(x, -1, -1) - 1
        G = G + F__ * f_n

        G = G * 2
        return G
    
        


if __name__ == '__main__':
    bal = BrownAlmostLinear(5)
    # x = bal.x_minimun_0
    print(bal.hessian(bal.x_minimun_0))
    print(bal.hessian(bal.x_minimun))
    # print(bal.func(bal.x_minimun_1))
    # print(bal.func(bal.x_0))
    # print(bal.grad(bal.x_0))
