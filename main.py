# coding=UTF-8
import numpy as np
import logging
import sys
from jqxss import GD_algorithm  # exact line search
from get_hessian import get_hessian
from newton import NewtonMethod
from inexact import InExactLineSearch
from quasi_newton import QuasiNewton
from brown_and_dennis import BrownAndDennis

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

# total_iter, x_k, loss = GD_algorithm(start_point=x_0,
#                                      func=func,
#                                      grad=grad,
#                                      G=G,
#                                      x_star=x_star,
#                                      epsilon=1e-8,
#                                      logger=logger)

# func_name = "example"
# total_iter, x_k, _ = NewtonMethod(start_point=x_0,
#                                   func=func,
#                                   grad=grad,
#                                   hessian=get_hessian(func_name),
#                                   x_star=x_star,
#                                   epsilon=1e-8,
#                                   max_iters=1e3,
#                                   method="hybrid wolfe interpolate22",
#                                   logger=logger)
# eps = ExtendedPowellSingular()
bad = BrownAndDennis(m=4)
total_iter, x_k, loss = InExactLineSearch(method="interpolate22 armijo",
                                          start_point=bad.x_0,
                                          func=bad.func,
                                          grad=bad.grad,
                                          x_star=x_star,
                                          f_minimun=bad.f_minimun,
                                          epsilon=1e-8,
                                          logger=logger)

# total_iter, x_k, loss = QuasiNewton(start_point=x_0,
#                                     func=func,
#                                     grad=grad,
#                                     x_star=x_star,
#                                     epsilon=1e-8,
#                                     max_iters=1e3,
#                                     method="bfgs strong_wolfe interpolate22",
#                                     logger=logger)

logger.info("***** Final Results *****")
logger.info("   迭代次数: " + str(total_iter))
logger.info("   函数调用次数: " + str(call_f))
logger.info("   迭代点的 x 值: " + str(x_k.reshape(1, -1)) + ", 函数值:" +
            str(bad.func(x_k)))
logger.info("   最优点的 x 值: " + str(x_star.reshape(1, -1)) + ", 最优函数值:" +
            str(bad.f_minimun))
