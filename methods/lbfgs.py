import numpy as np
from .criterion import criterion

S = np.array([])
Y = np.array([])


def getHg(g, hdiag):
    m, k = S.shape[0], S.shape[1]
    q = np.zeros(k + 1, dtype="float32")
    q[k] = g

    # init rho
    rho = np.zeros(k, dtype="float32")
    for i in range(k):
        rho[i] = 1 / (Y[i].T @ S[i])  # @?

    # loop 1
    alpha = np.zeros_like(S, dtype="float32")
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * S[i] @ q[i + 1]
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
          sigma=0.9,
          method="newton strong_wolfe simple",
          logger=None):

    global S
    global Y

    m = 5  # 10, 15
    x_k, loss, H = start_point, [], np.eye(len(start_point))
    if x_star is not None:
        f_minimun = func(x_star)

    for cnt_iter in range(int(max_iters)):
        g_k = grad(x_k).reshape(-1, 1)
        # ||g|| < eps 终止判断
        g_l2_norm = np.linalg.norm(g_k, ord=2)
        if g_l2_norm < epsilon:
            logger.info("g_l2_norm=" + str(g_l2_norm) + " < eps, 终止迭代")
            break

        if cnt_iter == 0:
            d_k = -g_k
            alpha, x_k_1 = criterion(method=method,
                                     x_k=x_k,
                                     d_k=d_k,
                                     func=func,
                                     grad=grad,
                                     m_max=20,
                                     rho=rho,
                                     eps=epsilon,
                                     sigma=sigma,
                                     logger=logger)
            # d_k = -alpha * g_k

        s_0 = x_k_1 - x_k
        y_0 = grad(x_k_1) - g_k
        H_0 = (s_0.T @ y_0) / (y_0.T @ y_0)

        if cnt_iter <= m:
            S.append(s_0)
            Y.append(y_0)
            d_k = -getHg(g_k, H_0)
        else:
            np.delete(S, 0)
            np.delete(Y, 0)
            S.append(s_0)
            Y.append(y_0)
            d_k = -getHg(g_k, H_0)

        x_k = x_k_1
        alpha, x_k_1 = criterion(method=method,
                                 x_k=x_k,
                                 d_k=d_k,
                                 func=func,
                                 grad=grad,
                                 m_max=20,
                                 rho=rho,
                                 eps=epsilon,
                                 sigma=sigma,
                                 logger=logger)

    return cnt_iter, x_k_1