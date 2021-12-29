# coding=UTF-8
import numpy as np
from .criterion import criterion
from functions import Trigonometric


def ConjugateGradient(start_point,
                      func,
                      grad,
                      hessian,
                      x_star,
                      f_minimun=None,
                      max_iters=1e3,
                      epsilon=1e-8,
                      rho=1e-4,
                      sigma=0.9,
                      init_alpha=0.5,
                      method="newton strong_wolfe simple",
                      logger=None):

    x_k, loss, beta = start_point, [], 0
    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))

        g_k = grad(x_k).reshape(-1, 1)

        # ||g||_2 < eps 终止判断
        g_k_l2norm = np.sqrt(g_k.T @ g_k)
        if g_k_l2norm < epsilon:
            logger.info("g_k_l2norm=" + str(g_k_l2norm) + " < " +
                        str(epsilon) + ", 终止迭代")
            break

        if cnt_iter == 0:
            d_k = -g_k

        gk_dk = g_k.T @ d_k

        if gk_dk > 0:
            d_k = -d_k / np.sqrt(d_k.T @ d_k)
        else:
            d_k = d_k / np.sqrt(d_k.T @ d_k)

        alpha, x_k = criterion(method=method,
                               x_k=x_k,
                               d_k=d_k,
                               func=func,
                               grad=grad,
                               m_max=20,
                               rho=rho,
                               eps=epsilon,
                               init_alpha=init_alpha,
                               sigma=sigma,
                               logger=logger)

        # if cnt_iter > 0:
        if cnt_iter > 0:
            if 'fr-prp' in method:
                beta_fr = (g_k.T @ g_k) / (g_k_1.T @ g_k_1)
                beta_prp = (g_k.T @ (g_k - g_k)) / (g_k_1.T @ g_k_1)
                if np.abs(beta_prp) < beta_fr:
                    beta = beta_prp
                elif beta_prp < -beta_fr:
                    beta = -beta_fr
                elif beta_prp > beta_fr:
                    beta = beta_fr
                else:
                    raise RuntimeError("unknown error at ConjugateGradient()")
            elif 'fr' in method:
                beta = (g_k.T @ g_k) / (g_k_1.T @ g_k_1)
            elif 'prp' in method:
                beta = (g_k.T @ (g_k - g_k_1)) / (g_k_1.T @ g_k_1)
                if 'prp+' in method:
                    beta = max(beta, 0)
            elif 'cd' in method:
                beta = -(g_k.T @ g_k) / (d_k_1.T @ g_k_1)
            elif 'dy' in method:
                beta = (g_k.T @ g_k) / (d_k_1.T @ (g_k - g_k_1))

        # |f(x_k_1) - f(x_k)| < eps 终止判断
        # if func == Trigonometric:
        if cnt_iter > 0:
            diff = np.fabs(func(x_k_1) - func(x_k))
            # if diff < epsilon:
            #     logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
            #                 str(np.fabs(func(x_k_1) - func(x_k))))
            #     break

            # if cnt_iter % 1000 == 0:
            #     logger.info("    当前迭代 " + str(cnt_iter))
            #     logger.info("    迭代点函数值 " + str(func(x_k_1)))
            #     diff = np.fabs(func(x_k_1) - func(x_k))
            #     logger.info("    |f(k) - f(k-1)| = " + str(diff))
            #     g_k_l2norm = np.sqrt(g_k.T @ g_k)
            #     logger.info("    ||g_k|| = " + str(g_k_l2norm))

        d_k_1 = d_k
        x_k_1 = x_k
        g_k_1 = g_k
        d_k = -g_k + beta * d_k

    return cnt_iter, x_k, g_k, diff