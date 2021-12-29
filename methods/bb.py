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
       init_alpha=0.5,
       sigma=0.4,
       method="newton strong_wolfe simple",
       logger=None):

    x_k, loss, M = start_point, [], 5
    alpha, delta, gamma, sigma = 1, 1, 1e-4, 0.4
    f_arr = np.ones(M, dtype="float32") * float('-inf')

    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))
        if cnt_iter == 0:
            g_k = grad(x_k).reshape(-1, 1)
        f_arr[cnt_iter - M * (cnt_iter // M)] = func(x_k)

        # ||g||_2 < eps 终止判断
        g_k_l2norm = np.sqrt(g_k.T @ g_k)
        if g_k_l2norm < epsilon:
            logger.info("g_k_l2norm=" + str(g_k_l2norm) + " < " +
                        str(epsilon) + ", 终止迭代")
            break
        if g_k_l2norm > 1:
            delta = 1.
        elif g_k_l2norm < 1e-5:
            delta = 1e5
        else:
            delta = 1. / g_k_l2norm

        if alpha <= epsilon or alpha >= 1 / epsilon:
            alpha = delta

        lambd = 1 / alpha

        # nonmonotone line search
        for _ in range(10):  # this loop can be enlarged
            if func(x_k - lambd * g_k) < (max(f_arr) -
                                          gamma * lambd * g_k.T @ g_k):
                break
            else:
                lambd *= sigma

        x_k_1 = x_k - lambd * g_k

        # |f(x_k_1) - f(x_k)| < eps 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        # if diff < epsilon:
        #     logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
        #                 str(np.fabs(func(x_k_1) - func(x_k))))
        #     break

        if (cnt_iter + 1) % 1000 == 0:
            logger.info("    当前迭代 " + str(cnt_iter))
            logger.info("    迭代点函数值 " + str(func(x_k_1)))
            diff = np.fabs(func(x_k_1) - func(x_k))
            logger.info("    |f(k) - f(k-1)| = " + str(diff))
            g_k_l2norm = np.sqrt(g_k.T @ g_k)
            logger.info("    ||g_k|| = " + str(g_k_l2norm))

        g_k_1 = grad(x_k_1)
        y = g_k_1 - g_k
        # alpha = -(g_k.T @ y_k) / (lambd * g_k.T @ g_k)
        s = x_k_1 - x_k
        alpha = (s.T @ y) / (s.T @ s)
        # alpha =  (y.T @ y) / (s.T @ y)
        x_k = x_k_1
        g_k = g_k_1

    return cnt_iter, x_k, g_k, diff
