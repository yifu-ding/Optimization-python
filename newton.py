import numpy as np
from inexact import criterion


# Newton 算法函数说明
def NewtonMethod(start_point,
                 func,
                 grad,
                 hessian,
                 x_star,
                 epsilon=1e-8,
                 max_iters=1e3,
                 method="damped wolfe interpolate22",
                 logger=None):
    # 参数说明
    x_k, loss = start_point, []

    for cnt_iter in range(int(max_iters)):
        logger.info("iter " + str(cnt_iter))
        g_k = grad(x_k).reshape(-1, 1)
        # G_k = hessian(x_k)
        G_k = hessian

        # newton 方向: d_k = -G^{-1}_k .* g_k

        if np.linalg.det(G_k) > 0:
            d_k = -np.dot(np.linalg.inv(G_k), g_k)
        elif np.linalg.det(G_k) < 0:
            d_k = np.dot(np.linalg.inv(G_k), g_k)
        else:
            d_k = -g_k
            logger.warning("Hessian 矩阵奇异, 不可用普通 Newton 方法求 d_k, 令 d_k = -g_k")
            # raise RuntimeError("det(G)==0")

        if "newton" in method:  # 普通 Newton 方法
            alpha = 1
            x_k_1 = x_k + alpha * d_k
        elif "damped" in method:  # 阻尼 Newton 方法
            alpha, x_k_1 = criterion(method,
                                     x_k,
                                     d_k,
                                     func,
                                     grad,
                                     m_max=20,
                                     logger=logger)
        elif "hybrid" in method:  # 混合 Newton 方法
            g_k_l2norm = np.linalg.norm(g_k, ord=2)
            d_k_l2norm = np.linalg.norm(d_k, ord=2)

            if np.dot(g_k.T, d_k) > (epsilon * g_k_l2norm * d_k_l2norm):
                d_k = -d_k  # d_k 不是下降方向，则取反方向
            elif np.abs(np.dot(g_k.T,
                               d_k)) <= (epsilon * g_k_l2norm * d_k_l2norm):
                d_k = -g_k  # 接近正交，改成最速下降
            # else:
            # raise NotImplementedError("应该不会走到这")
            alpha, x_k_1 = criterion(method,
                                     x_k,
                                     d_k,
                                     func,
                                     grad,
                                     m_max=20,
                                     logger=logger)

        elif "lm" in method:  # LM 方法
            raise NotImplementedError("有时间就写 LM 方法")
        else:
            raise NotImplementedError("未定义的 Newton 方法")

        loss_k = np.fabs(func(x_k_1) - func(x_star))
        loss.append(loss_k)
        logger.info("iter " + str(cnt_iter))
        logger.info("x_k=" + str(x_k.reshape(1, -1)))
        logger.info("loss_k=" + str(loss_k))

        # stopping criterion 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        logger.info("func(x_k_1)=" + str(func(x_k_1)))
        logger.info("func(x_k)=" + str(func(x_k)))
        logger.info("diff=" + str(diff))
        logger.info("")

        if diff < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
                        str(np.fabs(func(x_k_1) - func(x_k))))
            break

        x_k = x_k_1

    return cnt_iter, x_k, loss