# coding=UTF-8
import numpy as np
from .criterion import criterion


def InExactLineSearch(start_point,
                      func,
                      grad,
                      hessian=None,
                      x_star=None,
                      f_minimun=None,
                      max_iters=1e3,
                      epsilon=1e-8,
                      rho=1e-4,
                      sigma=0.9,
                      beta=0.5,
                      method=None,
                      logger=None):

    x_k, loss = start_point, []
    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))

        g_k = grad(x_k).reshape(-1, 1)  # 在 x_k 点处的函数导数值 g_k
        d_k = -g_k  # 最速下降方法的搜索方向

        alpha, x_k_1 = criterion(method=method,
                                 x_k=x_k,
                                 d_k=d_k,
                                 func=func,
                                 grad=grad,
                                 m_max=20,
                                 rho=rho,
                                 eps=epsilon,
                                 beta=beta,
                                 sigma=sigma,
                                 logger=logger)

        # |f(x_k_1) - f(x_k)| < eps 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        if diff < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " + str(diff))
            break

        x_k = x_k_1

    return cnt_iter, x_k_1