# coding=UTF-8
import numpy as np
import logging
import sys
from jqxss import GD_algorithm  # exact line search
from inexact import InExactLineSearch

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
    y = [1.5, 2.25, 2.625]

    ret = 0
    for i in range(3):
        i = i + 1
        r_i = y[i-1] - x[0] * (1 - np.power(x[1], i))
        ret = r_i * r_i + ret
    return ret

def grad(x):
    y = [1.5, 2.25, 2.625]

    ret = np.zeros((2,), dtype=np.float)
    for i in range(3):
        i = i + 1
        r_i = y[i-1] - x[0] * (1 - np.power(x[1], i))
        gr = np.zeros_like(ret)
        gr[0] = -(1 - np.power(x[1], i))
        gr[1] = x[0] * i * np.power(x[1], i - 1)
        ret = ret + 2 * r_i * gr
    return ret



# start point
x_0 = np.random.rand(2, 1)

# minima
x_star = np.array([3, 0.5])

print(func(x_star), grad(x_star))

# total_iter, x_k, loss = GD_algorithm(start_point=x_0,
#                                      func=func,
#                                      grad=grad,
#                                      G=G,
#                                      x_star=x_star,
#                                      epsilon=1e-8,
#                                      logger=logger)

total_iter, x_k, loss = InExactLineSearch(method="interpolate33 armijo",
                  start_point=x_0,
                  func=func,
                  grad=grad,
                  G=G,
                  x_star=x_star,
                  epsilon=1e-8,
                  logger=logger)

logger.info("***** Final Results *****")
logger.info("   迭代次数: " + str(total_iter))
logger.info("   函数调用次数: " + str(call_f))
logger.info("   迭代点的 x 值: " + str(x_k.reshape(1, -1)) + ", 函数值:" +
            str(func(x_k)))
logger.info("   最优点的 x 值: " + str(x_star.reshape(1, -1)) + ", 最优函数值:" +
            str(func(x_star)))
