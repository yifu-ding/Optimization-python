# coding=UTF-8
import numpy as np
from .criterion import criterion


def NewtonMethod(start_point,
                 func,
                 grad,
                 hessian,
                 x_star,
                 max_iters=1e3,
                 epsilon=1e-8,
                 rho=1e-4,
                 sigma=0.9,
                 method="newton strong_wolfe simple",
                 logger=None):

    x_k, loss = start_point, []

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))
        g_k = grad(x_k).reshape(-1, 1)
        G_k = hessian(x_k)

        # newton 方向: d_k = -G^{-1}_k .* g_k
        if np.linalg.det(G_k) > 0:
            d_k = -np.dot(np.linalg.inv(G_k), g_k)
        elif np.linalg.det(G_k) < 0:
            d_k = np.dot(np.linalg.inv(G_k), g_k)
        else:
            logger.warning("Hessian 矩阵奇异, 不可用普通 Newton 方法求 d_k, 令 d_k = -g_k")
            d_k = -g_k

        if "newton" in method:  # 普通 Newton 方法
            alpha = 1
            x_k_1 = x_k + alpha * d_k
        elif "damped" in method:  # 阻尼 Newton 方法
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
        elif "hybrid" in method:  # 混合 Newton 方法
            g_k_l2norm = np.linalg.norm(g_k, ord=2)
            d_k_l2norm = np.linalg.norm(d_k, ord=2)

            if np.dot(g_k.T, d_k) > (epsilon * g_k_l2norm * d_k_l2norm):
                d_k = -d_k  # d_k 不是下降方向，则取反方向
            elif np.abs(np.dot(g_k.T,
                               d_k)) <= (epsilon * g_k_l2norm * d_k_l2norm):
                d_k = -g_k  # 接近正交，改成最速下降

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

        elif "lm" in method:  # LM 方法
            v_0 = 1e-2
            while not np.all(np.linalg.eigvals(G_k) > 0):
                G_k += np.eye(G_k.shape[0]) * v_0
                v_0 *= 2
            d_k = -np.dot(np.linalg.inv(G_k), g_k)

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

        else:
            raise NotImplementedError("未定义的 Newton 方法")

        # |f(x_k_1) - f(x_k)| < eps 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        if diff < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " + str(diff))
            break

        x_k = x_k_1

    return cnt_iter, x_k