# coding=UTF-8
import numpy as np
import logging
import sys
import time
import argparse

from functions import BrownAndDennis, BrownAlmostLinear, Example, ExtendedPowellSingular
from methods import InExactLineSearch, NewtonMethod, QuasiNewton

# logger settings
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_name",
                        default="example",
                        type=str,
                        help="Name of the objective function.",
                        choices=[
                            "example", "brown_and_dennis",
                            "brown_almost_linear", "extended_powell_singular"
                        ])
    parser.add_argument("--stepsize_method",
                        default="simple",
                        type=str,
                        help="Method of getting stepsize.",
                        choices=['simple', 'interpolate22', 'interpolate33'])
    parser.add_argument(
        "--criterion_method",
        default="strong_wolfe",
        type=str,
        help="Criterion method.",
        choices=['armijo', 'goldstein', 'wolfe', 'strong_wolfe'])
    parser.add_argument("--opt_method",
                        default="newton",
                        type=str,
                        help="Optimization method.",
                        choices=[
                            'inexact', 'newton', 'damped', 'hybrid', 'lm',
                            'sr1', 'bfgs', 'dfp'
                        ])
    parser.add_argument("--max_iters",
                        default=1e3,
                        type=float,
                        help="Maximum iteration numbers.")
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--eps",
                        default=1e-8,
                        type=float,
                        help="Stopping criterion.")
    parser.add_argument("--m", default=20, type=int)

    args = parser.parse_args()

    if args.func_name == "brown_and_dennis":
        question = BrownAndDennis(m=args.m)
    elif args.func_name == "extended_powell_singular":
        question = ExtendedPowellSingular(m=args.m)
    elif args.func_name == "brown_almost_linear":
        question = BrownAlmostLinear(m=args.m)
    else:
        question = Example()

    start_time = time.process_time()

    if args.stepsize_method != "simple":
        logger.warning(
            "Unstable behavior due to unknown bug, please dont use it.")

    if args.opt_method == "inexact":
        total_iter, x_k = InExactLineSearch(method=args.stepsize_method +
                                            args.criterion_method,
                                            start_point=question.x_0,
                                            func=question.func,
                                            grad=question.grad,
                                            x_star=None,
                                            f_minimun=question.f_minimun,
                                            max_iters=args.max_iters,
                                            epsilon=args.eps,
                                            rho=args.rho,
                                            sigma=args.sigma,
                                            logger=logger)
    elif (args.opt_method == "newton") or (args.opt_method == "damped") or (
            args.opt_method == "hybrid") or (args.opt_method == "lm"):
        total_iter, x_k = NewtonMethod(start_point=question.x_0,
                                       func=question.func,
                                       grad=question.grad,
                                       hessian=question.hessian,
                                       x_star=question.x_star,
                                       max_iters=args.max_iters,
                                       epsilon=args.eps,
                                       rho=args.rho,
                                       sigma=args.sigma,
                                       method=args.stepsize_method +
                                       args.criterion_method + args.opt_method,
                                       logger=logger)
    elif (args.opt_method == "sr1") or (args.opt_method
                                        == "bfgs") or (args.opt_method
                                                       == "dfp"):
        total_iter, x_k = QuasiNewton(start_point=question.x_0,
                                      func=question.func,
                                      grad=question.grad,
                                      x_star=question.x_star,
                                      f_minimun=question.f_minimun,
                                      max_iters=args.max_iters,
                                      epsilon=args.eps,
                                      rho=args.rho,
                                      sigma=args.sigma,
                                      method=args.stepsize_method +
                                      args.criterion_method + args.opt_method,
                                      logger=logger)
    else:
        raise NotImplementedError("Optimization method is not implemented.")

    end_time = time.process_time()
    logger.info("***** Final Results *****")
    logger.info("   迭代次数(ite): " + str(total_iter))
    logger.info("   函数调用次数(feva): " + str(question.call_f))
    logger.info("   迭代点的 x 值: " + str(x_k.reshape(1, -1)) + ", 函数值:" +
                str(question.func(x_k)))
    logger.info("   最优函数值: " + str(question.f_minimun))
    logger.info("   CPU时间（ms）: " + str((end_time - start_time)))


if __name__ == "__main__":
    main()
