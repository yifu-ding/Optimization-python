# coding=UTF-8
# 精确线性搜索 exact line search
import numpy as np


def GD_algorithm(start_point,
                 func,
                 grad,
                 hessian,
                 x_star,
                 epsilon=1e-8,
                 max_iters=1e3,
                 logger=None):
    """
    :param start_point: start point of GD algorithm
    :param func: function defined in the question
    :param grad: gradient of the function
    :param epsilon: stopping criterion
    """
    # 初始化设置: 初始点 x_k=start_point,
    #            最大迭代次数 max_iters,
    #            函数收敛情况 loss
    x_k, loss = start_point, []

    for cnt_iter in range(int(max_iters)):
        # 在 x_k 点处的函数导数值 g_k
        g_k = grad(x_k).reshape(-1, 1)

        # d_k 是点 x_k 使得 func 下降的方向
        # min func(x_k + alpha_k * d_k) -> alpha_k
        alpha_k = -np.dot(g_k.T, g_k).squeeze() / (np.dot(
            g_k.T, np.dot(G, g_k))).squeeze()

        x_k_1 = x_k + alpha_k * g_k

        # 计算此时迭代点 x_k_1 与最优解 x_star 的 loss
        loss_k = np.fabs(func(x_k_1) - func(x_star))
        loss.append(loss_k)
        logger.info("iter " + str(cnt_iter))
        logger.info("x_k=" + str(x_k.reshape(1, -1)))
        logger.info("loss_k=" + str(loss_k))
        logger.info("")

        # stopping criterion 终止判断
        if np.fabs(func(x_k_1) - func(x_k)) < epsilon:
            logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
                        str(np.fabs(func(x_k_1) - func(x_k))))
            break

        x_k = x_k_1

    return cnt_iter, x_k_1, loss