# coding=UTF-8
import numpy as np
from .criterion import criterion


def QuasiNewton(start_point,
                func,
                grad,
                hessian,
                x_star,
                f_minimun,
                max_iters=1e3,
                epsilon=1e-8,
                rho=1e-4,
                beta=0.5,
                sigma=0.9,
                method="sr1 wolfe interpolate22",
                logger=None):

    x_k, loss, H = start_point, [], np.eye(len(start_point))
    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))

        g_k = grad(x_k).reshape(-1, 1)
        # ||g|| < eps 终止判断
        if np.linalg.norm(g_k, ord=2) < epsilon:
            # logger.info("g_k L2 norm < eps, 终止迭代")
            break
        d_k = -np.dot(H, g_k)
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
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
                        str(np.fabs(func(x_k_1) - func(x_k))))
            break

        s = x_k_1 - x_k
        y = grad(x_k_1).reshape(-1, 1) - g_k
        if "sr1" in method:
            tmp = s - np.dot(H, y)
            H = H + np.dot(tmp, tmp.T) / np.dot(tmp.T, y)
        elif "dfp" in method:
            H = H + np.dot(s, s.T) / np.dot(s.T, y) - np.dot(
                np.dot(np.dot(H, y), y.T), H) / np.dot(np.dot(y.T, H), y)
        elif "bfgs" in method:
            h1 = 1 + np.dot(np.dot(y.T, H), y) / np.dot(y.T, s)
            h2 = np.dot(s, s.T) / np.dot(y.T, s)
            # import pdb
            # pdb.set_trace()
            h3 = np.dot(np.dot(s, y.T), H) + np.dot(np.dot(H, y), s.T)
            H = H + h1 * h2 - h3 / np.dot(y.T, s)
        else:
            raise NotImplementedError("未定义的 Quasi-Newton 方法")

        x_k = x_k_1

    return cnt_iter, x_k