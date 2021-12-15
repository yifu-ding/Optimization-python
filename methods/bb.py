# coding=UTF-8
import numpy as np
from .criterion import criterion


def BB(start_point,
       func,
       grad,
       hessian,
       x_star,
       f_minimun=None,
       max_iters=1e3,
       epsilon=1e-8,
       rho=1e-4,
       sigma=0.9,
       method="newton strong_wolfe simple",
       logger=None):

    x_k, loss, beta = start_point, [], 0

    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))
        g_k = grad(x_k).reshape(-1, 1)
        # ||g|| < eps 终止判断
        g_l2_norm = np.linalg.norm(g_k, ord=2)
        if g_l2_norm < epsilon:
            logger.info("g_l2_norm=" + str(g_l2_norm) + " < eps, 终止迭代")
            break

        if cnt_iter == 0:
            d_k = -g_k
            alpha, x_k_1 = criterion(method=method,
                                     x_k=x_k,
                                     d_k=d_k,
                                     func=func,
                                     grad=grad,
                                     m_max=20,
                                     rho=rho,
                                     eps=epsilon,
                                     sigma=sigma,
                                     logger=logger)

    return cnt_iter, x_k
