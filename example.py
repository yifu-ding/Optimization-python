import numpy as np


class Example:
    def __init__(self):
        self.G = np.array([[10, -9], [-9, 10]], dtype="float32")
        self.b = np.array([4, -15], dtype="float32").reshape([-1, 1])
        self.call_f = 0
        # self.x_0 = np.squeeze(np.random.rand(2, 1), axis=-1)
        self.x_0 = np.random.rand(2, 1).reshape([-1, 1])
        self.x_star = -np.dot(np.linalg.inv(self.G), self.b)
        self.f_minimun = self.func(self.x_star)

    def func(self, x):
        self.call_f += 1
        return 0.5 * np.dot(x.T, np.dot(self.G, x)).squeeze() + np.dot(
            self.b.T, x).squeeze()

    def grad(self, x):
        # import pdb
        # pdb.set_trace()
        return np.dot(self.G, x) + self.b

    def hessian(self, x):
        return self.G + self.G.T