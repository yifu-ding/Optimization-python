# coding=UTF-8
import numpy as np
import logging
import sys
import time
import argparse

from functions import BrownAndDennis, BrownAlmostLinear, Example, ExtendedPowellSingular, Penalty, Trigonometric, ExtendedRosenbrock
from methods import InExactLineSearch, NewtonMethod, QuasiNewton, LBFGS, ConjugateGradient, BB

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
                            "brown_almost_linear", "extended_powell_singular",
                            "penalty", "trigonometric", "extended_rosenbrock"
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
                            'sr1', 'bfgs', 'dfp', 'lbfgs', 'fr', 'prp', 'prp+',
                            'bb'
                        ])
    parser.add_argument("--max_iters",
                        default=1e3,
                        type=float,
                        help="Maximum iteration numbers.")
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--eps",
                        default=1e-8,
                        type=float,
                        help="Stopping criterion.")
    parser.add_argument("--m", default=20, type=int)

    args = parser.parse_args()

    methods = {
        'inexact': InExactLineSearch,
        'newton': NewtonMethod,
        'damped': NewtonMethod,
        'hybrid': NewtonMethod,
        'lm': NewtonMethod,
        'sr1': QuasiNewton,
        'bfgs': QuasiNewton,
        'dfp': QuasiNewton,
        'lbfgs': LBFGS,
        'fr': ConjugateGradient,
        'prp': ConjugateGradient,
        'prp+': ConjugateGradient,
        'bb': BB,
    }

    if args.stepsize_method != "simple":
        logger.warning(
            "Unstable behavior due to unknown bug, please dont use it.")

    questions = {
        'brown_and_dennis': BrownAndDennis,
        'extended_powell_singular': ExtendedPowellSingular,
        'brown_almost_linear': BrownAlmostLinear,
        'penalty': Penalty,
        'trigonometric': Trigonometric,
        'extended_rosenbrock': ExtendedRosenbrock,
        'example': Example
    }

    question = questions[args.func_name](m=args.m)

    start_time = time.process_time()

    total_iter, x_k = methods[args.opt_method](
        start_point=question.x_0,
        func=question.func,
        grad=question.grad,
        hessian=question.hessian,
        x_star=question.x_star,
        f_minimun=question.f_minimun,
        max_iters=args.max_iters,
        epsilon=args.eps,
        beta=args.beta,
        rho=args.rho,
        sigma=args.sigma,
        method=args.stepsize_method + args.criterion_method + args.opt_method,
        logger=logger)

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
