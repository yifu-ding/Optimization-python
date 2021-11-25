# coding=UTF-8
import numpy as np
from interpolate import get_alpha


def criterion(method, x_k, d_k, func, grad, m_max, logger):
    # TODO: 参数说明
    # TODO: 初始化说明
    beta = 0.5  # armijo 变体方法，步长的初始值
    rho = 1e-4  # refer to book P21
    eps = 1e-8
    sigma = 0.9  # 越小越接近精确线搜索
    alpha = np.array([0.1], dtype="float32").reshape(-1, 1)  # init
    if "interpolate33" in method:
        alpha = np.array([0.1, 0.5], dtype="float32").reshape(-1, 1)

    for _ in range(int(m_max)):
        alpha = get_alpha(x_k=x_k,
                          d_k=d_k,
                          alpha=alpha,
                          beta=beta,
                          func=func,
                          grad=grad,
                          m=_,
                          method=method)

        f_k_1 = func(x_k + np.squeeze(alpha[0] * d_k, axis=-1))
        f_k = func(x_k)
        g_k = grad(x_k)

        gk_dk_alpha = np.dot(grad(x_k).T, d_k) * alpha[0]
        gk1_dk = np.dot(grad(x_k + alpha[0] * d_k).T, d_k)

        satisfy = False
        g_k_l2norm = np.linalg.norm(g_k, ord=2)
        if g_k_l2norm < eps:
            satisfy = True

        if "armijo" in method:
            import pdb
            pdb.set_trace()
            if f_k_1 <= (f_k + rho * gk_dk_alpha):
                satisfy = True
            else:
                satisfy = False
        elif "goldstein" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha) and f_k_1 >= (
                    f_k + (1 - rho) * gk_dk_alpha):
                satisfy = True
            else:
                satisfy = False
        elif "wolfe" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha) and gk1_dk >= (
                    sigma * np.dot(g_k.T, d_k)):
                satisfy = True
            else:
                satisfy = False
        elif "strong_wolfe" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha
                         ) and np.abs(gk1_dk) <= -(sigma * np.dot(g_k.T, d_k)):
                satisfy = True
            else:
                satisfy = False
        else:
            raise NotImplementedError("method " + str(method) +
                                      " not implemented")
        if satisfy:
            break

    x_k_1 = x_k + alpha * d_k
    return alpha, x_k_1


def InExactLineSearch(method,
                      start_point,
                      func,
                      grad,
                      x_star=None,
                      f_minimun=None,
                      epsilon=1e-8,
                      max_iters=1e3,
                      logger=None):

    x_k, loss = start_point, []
    if x_star is None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        logger.info("iter " + str(cnt_iter))

        g_k = grad(x_k).reshape(-1, 1)  # 在 x_k 点处的函数导数值 g_k
        d_k = -g_k  # 最速下降方法的搜索方向
        alpha, x_k_1 = criterion(method=method,
                                 x_k=x_k,
                                 d_k=d_k,
                                 func=func,
                                 grad=grad,
                                 m_max=20,
                                 logger=logger)
        logger.info("当前步长 alpha=" + str(alpha))
        logger.info("最速下降方法搜索方向 d_k=" + str(d_k))
        logger.info("当前迭代点 x_k_1=" + str(x_k_1))

        # 计算此时迭代点 x_k_1 与最优解 x_star 的 loss
        loss_k = np.fabs(func(x_k_1) - f_minimun)
        loss.append(loss_k)
        logger.info("loss_k=" + str(loss_k))
        logger.info("")

        # stopping criterion 终止判断
        if np.fabs(func(x_k_1) - func(x_k)) < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
                        str(np.fabs(func(x_k_1) - func(x_k))))
            break

        x_k = x_k_1

    return cnt_iter, x_k_1, loss