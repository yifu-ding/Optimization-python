import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Brown and Dennis function
# dimentions
n = 4

# start point
x_0 = np.array([25, 5, -5, -1], dtype="float32")

# minima
f_minima, m_minima = 85822.2, 20

# function definition
t = lambda t: t / 5.
func = lambda t_l, x: (x[0] + t_l * x[1] - np.exp(t_l))**2 + (x[2] + x[
    3] * np.sin(t_l) - np.cos(t_l))**2
grad = lambda t_l, x: 0  # TODO


# GD algorithm
def GD_algorithm(start_point, func, grad, epsilon=1e-8):
    """
    :param start_point: start point of GD algorithm
    :param func: function defined in the question
    :param grad: gradient of the function
    :param epsilon: stopping criterion
    """
    x_0, max_iters, = start_point, 1e3

    for cnt_iter in trange(int(max_iters), desc="Iter"):
        # stopping criterion
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:
            break

    return