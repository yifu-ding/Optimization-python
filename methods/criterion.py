# coding=UTF-8
import numpy as np
from .get_stepsize import get_alpha


def criterion(method,
              x_k,
              d_k,
              func,
              grad,
              m_max,
              rho,
              eps,
              init_alpha=0.5,
              sigma=0,
              logger=None):
    alpha = np.array(init_alpha, dtype="float32").reshape(-1, 1)
    if "interpolate33" in method:
        alpha = np.array([1., 0.5], dtype="float32").reshape(-1, 1)

    f_k = func(x_k)
    g_k = grad(x_k)

    g_k_l2norm = np.sqrt(g_k.T @ g_k)
    if g_k_l2norm < eps:
        # logger.info("g_k L2 < eps, 终止迭代")
        return alpha, x_k

    gk1_dk = np.dot(grad(x_k + alpha[0] * d_k).T, d_k)

    for _ in range(int(m_max)):
        alpha_t = get_alpha(x_k=x_k,
                            d_k=d_k,
                            alpha=alpha,
                            init_alpha=init_alpha,
                            func=func,
                            grad=grad,
                            m=_,
                            method=method)
        if alpha == alpha_t:
            alpha = alpha_t
            # logger.info("步长不改变，停止循环")
            break
        alpha = alpha_t

        alpha_abs = np.abs(alpha[0])
        if alpha_abs < eps:
            # logger.info("步长太小 alpha=" + str(alpha))
            alpha = np.array([0.1 * init_alpha],
                             dtype="float32").reshape(-1, 1)  # init
            break

        f_k_1 = func(x_k + alpha[0] * d_k)
        gk_dk_alpha = np.squeeze(np.dot(grad(x_k).T, d_k) * alpha[0], axis=-1)

        satisfy = False
        if "armijo" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha):
                satisfy = True
        elif "goldstein" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha) and f_k_1 >= (
                    f_k + (1 - rho) * gk_dk_alpha):
                satisfy = True
        elif "wolfe" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha) and gk1_dk >= (
                    sigma * np.dot(g_k.T, d_k)):
                satisfy = True
        elif "strong_wolfe" in method:
            if f_k_1 <= (f_k + rho * gk_dk_alpha
                         ) and np.abs(gk1_dk) <= -(sigma * np.dot(g_k.T, d_k)):
                satisfy = True
        else:
            raise NotImplementedError("Method " + str(method) +
                                      " not implemented")
        if satisfy:
            # logger.info("在第" + str(_) + "次迭代，步长满足准则")
            break
    # if _ == m_max - 1:
    #     # raise RuntimeError("步长 alpha=" + str(alpha) + "未满足准则，但达到迭代次数")
    #     logger.info("步长 alpha=" + str(alpha) + "未满足准则，但达到迭代次数")

    x_k_1 = x_k + alpha[0] * d_k
    return alpha, x_k_1
