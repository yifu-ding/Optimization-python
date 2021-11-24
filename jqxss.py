#coding=UTF-8
# 精确线性搜索
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import logging
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

# function definition
G = np.array([[10, -9], [-9, 10]], dtype="float32")
b = np.array([4, -15], dtype="float32").reshape([-1, 1])
call_f = 0


def func(x):
    global call_f
    call_f += 1
    return 0.5 * np.dot(x.T, np.dot(G, x)).squeeze() + np.dot(b.T, x).squeeze()


grad = lambda x: np.dot(G, x) + b

# start point
x_0 = np.random.rand(2, 1)

# minima
x_star = -np.dot(np.linalg.inv(G), b)


# GD algorithm
def GD_algorithm(start_point, func, grad, epsilon=1e-8):
    """
    :param start_point: start point of GD algorithm
    :param func: function defined in the question
    :param grad: gradient of the function
    :param epsilon: stopping criterion
    """
    # 初始化设置: 初始点 x_0, 最大迭代次数 max_iters, 函数收敛情况 loss
    x_0, max_iters, loss = start_point, 1e3, []
    x_k = x_0
    for cnt_iter in trange(int(max_iters), desc="Iter"):
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


total_iter, x_k, loss = GD_algorithm(x_0, func, grad)
logger.info("***** Final Results *****")
logger.info("   迭代次数: " + str(total_iter))
logger.info("   函数调用次数: " + str(call_f))
logger.info("   迭代点的 x 值: " + str(x_k.reshape(1, -1)) + ", 函数值:" +
            str(func(x_k)))
logger.info("   最优点的 x 值: " + str(x_star.reshape(1, -1)) + ", 最优函数值:" +
            str(func(x_star)))
