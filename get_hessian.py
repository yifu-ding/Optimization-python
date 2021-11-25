# coding=UTF-8

import numpy as np


def get_hessian(func_name="extended_powell_singular"):
    if func_name == "example":
        G = np.array([[10, -9], [-9, 10]], dtype="float32")
        return G + G.T
    elif func_name == "extended_powell_singular":
        return G
    elif func_name == "":
        return
    elif func_name == "":
        return
    else:
        raise NotImplementedError("未定义目标函数")
    return G