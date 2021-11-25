# coding=UTF-8
import numpy as np
from inexact import criterion


def QuasiNewton(start_point,
                func,
                grad,
                x_star,
                logger,
                epsilon=1e-8,
                max_iters=1e3,
                method="sr1 wolfe interpolate22"):
    # 初始化
    x_k, loss, H = start_point, [], np.eye(len(start_point))

    for cnt_iter in range(int(max_iters)):
        logger.info("iter " + str(cnt_iter))
        g_k = grad(x_k).reshape(-1, 1)
        # 终止条件检测
        if np.linalg.norm(g_k, ord=2) < epsilon:
            break
        d_k = -np.dot(H, g_k)
        alpha, x_k_1 = criterion(method,
                                 x_k,
                                 d_k,
                                 func,
                                 grad,
                                 m_max=20,
                                 logger=logger)

        loss_k = np.fabs(func(x_k_1) - func(x_star))
        loss.append(loss_k)
        logger.info("iter " + str(cnt_iter))
        logger.info("x_k=" + str(x_k.reshape(1, -1)))
        logger.info("loss_k=" + str(loss_k))

        # stopping criterion 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        logger.info("")
        if diff < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
                        str(np.fabs(func(x_k_1) - func(x_k))))
            break

        s = x_k_1 - x_k
        y = grad(x_k_1) - g_k
        if "sr1" in method:
            tmp = s - H * y
            H = H + (tmp * tmp.T) / (tmp.T * y)
        elif "dfp" in method:
            H = H + (s * s.T) / (s.T * y) - (H * y * y.T * H) / (y.T * H * y)
            H[0, 1] = H[1, 0] = 0
        elif "bfgs" in method:
            H = H + (1 + (y.T * H * y) / (y.T * s)) * (s * s.T) / (y.T * s) - (
                (s * y.T * H + H * y * s.T) / (y.T * s))
        else:
            raise NotImplementedError("未定义的拟牛顿方法")

        x_k = x_k_1

    return cnt_iter, x_k, loss