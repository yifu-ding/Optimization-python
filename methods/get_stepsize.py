# coding=UTF-8
import numpy as np


# 两点两次插值 2-Point Quadric Intropolation
def interpolate22(func, x_k, g_k, d_k, alpha):
    f_k = func(x_k)  # phi(0)
    f_k_1 = func(x_k + alpha * d_k)  # phi(alpha_0)
    gk_dk = np.dot(g_k.T, d_k)  # phi'(0)
    alpha = -(gk_dk * alpha**2) / (2 * (f_k_1 - f_k - gk_dk * alpha))
    return alpha


# 三点三次插值 3-Point Cubic Intropolation
def interpolate33(func, x_k, g_k, d_k, alpha_0, alpha_1):
    f_k = func(x_k)  # phi(0)
    f_k_0 = func(x_k + alpha_0 * d_k)  # phi(alpha_0)
    f_k_1 = func(x_k + alpha_1 * d_k)  # phi(alpha_1)

    gk_dk = np.dot(g_k.T, d_k)  # phi'(0)
    # 三次插值函数 p(x) = ax^3 + bx^2 + cx + d
    #        导数 p'(x) = 3ax^2 + 2bx + c
    # 由插值条件 d = phi(0), c = phi'(0), 解关于 a, b 的线性方程组
    has_real_root = False
    try:
        if alpha_0 != 0 and alpha_1 != 0 and alpha_1 != alpha_0:
            mat_0 = np.array(
                [[alpha_0**2, -alpha_1**2], [-alpha_0**3, alpha_1**3]],
                dtype="float32")
            mat_0 = np.squeeze(mat_0, axis=-1)
            mat_1 = np.array(
                [f_k_1 - f_k - gk_dk * alpha_1, f_k_0 - f_k - gk_dk * alpha_0],
                dtype="float32").reshape([-1, 1])
            tmp = alpha_0**2 * alpha_1**2 * (alpha_1 - alpha_0)
            a_b = np.dot(mat_0, mat_1) / tmp
            assert a_b.shape == (2, 1)
            # 由 p'(x) = 3ax^2 + 2bx + c = 0, 解 x
            a_b[0] = a_b[0] * 3
            a_b[1] = a_b[1] * 2
            coeff = np.append(a_b, gk_dk)
            r = np.roots(coeff)
            if np.isreal(r).sum() == r.shape[0]:
                alpha_2 = np.max(r)
                has_real_root = True
    except:
        pass

    if not has_real_root:
        logger.info("方程的解不是实数，改用两点两次插值")
        alpha_2 = interpolate22(func, x_k, g_k, d_k, alpha_1)
    return np.array([alpha_2, alpha_1], dtype="float32").reshape(-1, 1)


def get_alpha(x_k, d_k, alpha, beta, func, grad, m, method="simple armijo"):
    if "simple" in method:  # armijo 的变体 beta**m
        return np.array([beta**m], dtype="float32").reshape(-1, 1)

    g_k = grad(x_k)
    if "interpolate22" in method:
        return interpolate22(func, x_k, g_k, d_k, alpha)
    elif "interpolate33" in method:
        assert len(alpha) == 2
        alpha_0 = alpha[0]
        alpha_1 = alpha[1]
        return interpolate33(func, x_k, g_k, d_k, alpha_0, alpha_1)
    else:
        raise NotImplementedError("未定义的求步长方法")
