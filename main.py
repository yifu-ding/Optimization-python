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
from extended_powell_singular import ExtendedPowellSingular
from example import Example

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

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
# question = ExtendedPowellSingular(m=4)
question = BrownAndDennis(m=20)
# question = Example()
total_iter, x_k, loss = InExactLineSearch(method="interpolate22 wolfe",
                                          start_point=question.x_0,
                                          func=question.func,
                                          grad=question.grad,
                                          x_star=None,
                                          f_minimun=question.f_minimun,
                                          epsilon=1e-8,
                                          logger=logger)

# total_iter, x_k, loss = QuasiNewton(start_point=question.x_0,
#                                     func=question.func,
#                                     grad=question.grad,
#                                     x_star=question.x_star,
#                                     f_minimun=question.f_minimun,
#                                     epsilon=1e-8,
#                                     max_iters=1e3,
#                                     method="sr1 wolfe simple",
#                                     logger=logger)

logger.info("***** Final Results *****")
logger.info("   迭代次数: " + str(total_iter))
logger.info("   函数调用次数: " + str(question.call_f))
logger.info("   迭代点的 x 值: " + str(x_k.reshape(1, -1)) + ", 函数值:" +
            str(question.func(x_k)))
logger.info("   最优函数值:" + str(question.f_minimun))
