import numpy as np


def quadratic_func(x: np.ndarray,t, eval_hessian: bool = False):
    # x=x0, y=x1, z=x2
    f_x = t * (x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2) - np.log(x[0]) - np.log(x[1]) - np.log(x[2])
    x_deriv = 2 * t * x[0] - 1 / x[0]
    y_deriv = 2 * t * x[1] - 1 / x[1]
    z_deriv = 2 * t * (x[2] + 1) - 1 / x[2]

    g_x = np.array([x_deriv,y_deriv,z_deriv])
    if eval_hessian:
        h_x = np.diag([2 * t + 1/((x[0])**2), 2 * t + 1/((x[1])**2), 2 * t + 1/((x[2])**2)])
        return f_x, g_x, h_x
    return f_x, g_x


def linear_func(x: np.ndarray, t, eval_hessian: bool = False):
    # x=x0, y=x1
    f_x = -t*x[0]-t*x[1]-np.log(x[0]+x[1]-1)-np.log(1-x[1])-np.log(2-x[0])-np.log(x[1])
    x_deriv = -t - 1 / (x[0] + x[1] - 1) + 1 / (2 - x[0])
    y_deriv = -t - 1 / (x[0] + x[1] - 1) + 1 / (1 - x[1]) - 1 / x[1]
    g_x = np.array([x_deriv, y_deriv])
    if eval_hessian:
        h_x = np.diag([1/((2-x[0])**2), 1/((1-x[1])**2)+1/(x[1]**2)])
        h_x += 1/((x[0]+x[1]-1)**2)
        return f_x, g_x, h_x
    return f_x, g_x

# def quadratic_func(x: np.ndarray, eval_hessian: bool = False):
#     # x=x0, y=x1, z=x2
#     f_x = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
#     g_x = np.array([2 * x[0],
#                     2 * x[1],
#                     2 * x[2] + 2])
#     if eval_hessian:
#         h_x = np.array([2, 0, 0,
#                         0, 2, 0,
#                         0, 0, 2])
#         return f_x, g_x, h_x
#     return f_x, g_x
#
#
# def linear_func(x: np.ndarray, eval_hessian: bool = False):
#     # x=x0, y=x1
#     f_x = -x[0]-x[1]
#     g_x = np.array([-1, -1])
#     if eval_hessian:
#         h_x = np.zeros((len(x), len(x)))
#         return f_x, g_x, h_x
#     return f_x, g_x


