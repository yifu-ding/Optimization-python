# coding=UTF-8
import numpy as np
from inexact import criterion


def QuasiNewton(start_point,
                func,
                grad,
                x_star,
                f_minimun,
                logger,
                epsilon=1e-8,
                max_iters=1e3,
                method="sr1 wolfe interpolate22"):
    # 初始化
    x_k, loss, H = start_point, [], np.eye(len(start_point))
    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        # logger.info("iter " + str(cnt_iter))

        g_k = grad(x_k).reshape(-1, 1)
        # 终止条件检测
        if np.linalg.norm(g_k, ord=2) < epsilon:
            # logger.info("g_k L2 norm < eps, 终止迭代")
            break
        d_k = -np.dot(H, g_k)
        alpha, x_k_1 = criterion(method,
                                 x_k,
                                 d_k,
                                 func,
                                 grad,
                                 m_max=20,
                                 logger=logger)

        # loss_k = np.fabs(func(x_k_1) - f_minimun)
        # loss.append(loss_k)
        # logger.info("x_k=" + str(x_k.reshape(1, -1)))
        # logger.info("loss_k=" + str(loss_k))

        # stopping criterion 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        # logger.info("diff=" + str(diff))
        # logger.info("")
        if diff < epsilon:
            # logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
            #             str(np.fabs(func(x_k_1) - func(x_k))))
            break

        s = x_k_1 - x_k
        y = grad(x_k_1) - g_k
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
            raise NotImplementedError("未定义的拟牛顿方法")

        x_k = x_k_1

    return cnt_iter, x_k, loss