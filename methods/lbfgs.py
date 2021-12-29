import numpy as np
from .criterion import criterion

S = []
Y = []


def getHg(g, hdiag):
    m = len(S)
    q = np.zeros((m + 1, S[0].shape[0], 1), dtype="float32")
    q[m] = g

    # init rho
    rho = np.zeros(m, dtype="float32")
    for i in range(m):
        rho[i] = 1 / (S[i].T @ Y[i])

    # loop 1
    alpha = np.zeros_like(S, dtype="float32")
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * (S[i].T @ q[i + 1])
        q[i] = q[i + 1] - alpha[i] * Y[i]

    # multi init Hessian
    r = hdiag * q[0]

    # loop 2
    beta = np.zeros_like(S, dtype="float32")
    for i in range(m):
        beta[i] = rho[i] * Y[i].T @ r
        r += S[i] * (alpha[i] - beta[i])
    return r


def LBFGS(start_point,
          func,
          grad,
          hessian,
          x_star,
          f_minimun=None,
          max_iters=1e3,
          epsilon=1e-8,
          rho=1e-4,
          init_alpha=0.5,
          sigma=0.9,
          method="newton strong_wolfe simple",
          logger=None):

    global S
    global Y

    M = 5  # 10, 15

    x_k, loss, H, m_max = start_point, [], np.eye(len(start_point)), 50

    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        g_k = grad(x_k).reshape(-1, 1)

        # ||g|| < eps 终止判断
        g_k_l2norm = np.sqrt(g_k.T @ g_k)
        if g_k_l2norm < epsilon:
            logger.info("g_k_l2norm=" + str(g_k_l2norm) + " < " +
                        str(epsilon) + ", 终止迭代")
            break

        if cnt_iter == 0:
            d_k = -g_k
            alpha, x_k_1 = criterion(method=method,
                                     x_k=x_k,
                                     d_k=d_k,
                                     func=func,
                                     grad=grad,
                                     m_max=m_max,
                                     rho=rho,
                                     eps=epsilon,
                                     init_alpha=init_alpha,
                                     sigma=sigma,
                                     logger=logger)
            alpha_abs = np.abs(alpha[0])

            if alpha_abs < epsilon:
                alpha = np.array([0.1 * init_alpha],
                                 dtype="float32").reshape(-1, 1)  # init
                logger.info("步长太小 重新选取 alpha=" + str(alpha))
                break

        s_0 = x_k_1 - x_k
        y_0 = grad(x_k_1) - g_k
        H_0 = (s_0.T @ y_0) / (y_0.T @ y_0)

        if cnt_iter <= M:
            S.append(s_0)
            Y.append(y_0)
            d_k = -getHg(g_k, H_0)
        else:
            S.pop(0)  # np.delete(S, 0)
            Y.pop(0)  # np.delete(Y, 0)
            S.append(s_0)
            Y.append(y_0)
            d_k = -getHg(g_k, H_0)

        x_k = x_k_1
        alpha, x_k_1 = criterion(method=method,
                                 x_k=x_k,
                                 d_k=d_k,
                                 func=func,
                                 grad=grad,
                                 m_max=m_max,
                                 rho=rho,
                                 eps=epsilon,
                                 init_alpha=init_alpha,
                                 sigma=sigma,
                                 logger=logger)
        # print("步长=" + str(alpha))

        # |f(x_k_1) - f(x_k)| < eps 终止判断
        diff = np.fabs(func(x_k_1) - func(x_k))
        # if diff < epsilon:
        #     logger.info("达到终止条件: func(x_k_1) - func(x_k) = " +
        #                 str(np.fabs(func(x_k_1) - func(x_k))))
        #     break

        if (cnt_iter + 1) % 1000 == 0:
            logger.info("    当前迭代 " + str(cnt_iter))
            logger.info("    迭代点函数值 " + str(func(x_k_1)))
            diff = np.fabs(func(x_k_1) - func(x_k))
            logger.info("    |f(k) - f(k-1)| = " + str(diff))
            g_k_l2norm = np.sqrt(g_k.T @ g_k)
            logger.info("    ||g_k|| = " + str(g_k_l2norm))

    return cnt_iter, x_k_1, g_k, diff
