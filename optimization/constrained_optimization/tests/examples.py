import numpy as np


def f_calc(x: np.ndarray, Q: np.ndarray):
    f_x = x.T.dot(Q).dot(x)
    return f_x


def f_calc_d1(x: np.ndarray, eval_hessian: bool = False):
    Q = np.array([[1, 0],
                  [0, 1]])
    f_x = f_calc(x, Q)
    g_x = 2 * Q.dot(x)
    if eval_hessian:
        h_x = 2 * Q
        return f_x, g_x, h_x
    return f_x, g_x


def f_calc_d2(x: np.ndarray, eval_hessian: bool = False):
    Q = np.array([[1, 0],
                  [0, 100]])
    f_x = f_calc(x, Q)
    g_x = 2 * Q.dot(x)
    if eval_hessian:
        h_x = 2 * Q
        return f_x, g_x, h_x
    return f_x, g_x


def f_calc_d3(x: np.ndarray, eval_hessian: bool = False):
    q1 = np.array([[np.sqrt(3) / 2, -0.5],
                   [0.5, np.sqrt(3) / 2]]).T
    q2 = np.array([[100, 0],
                   [0, 1]])
    q3 = np.array([[np.sqrt(3) / 2, -0.5],
                   [0.5, np.sqrt(3) / 2]])
    Q = (q1.dot(q2)).dot(q3)
    f_x = f_calc(x, Q)
    g_x = 2 * Q.dot(x)
    if eval_hessian:
        h_x = 2 * Q
        return f_x, g_x, h_x
    return f_x, g_x


def rosenbrock_func(x: np.ndarray, eval_hessian: bool = False):
    f_x = 100.0 * ((x[1] - x[0] ** 2) ** 2) + ((1 - x[0]) ** 2)
    g_x = np.array([-400.0 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                    200.0 * (x[1] - x[0] ** 2)])
    if eval_hessian:
        h_x = np.array([[-400.0 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]],  # Check if it is 2-2 or 2+2
                        [-400 * x[0], 200]])
        return f_x, g_x, h_x
    return f_x, g_x


def linear_func(x: np.ndarray, eval_hessian: bool = False):
    a = np.random.randint(1, 9, x.shape)
    f_x = a.T.dot(x)
    g_x = a.T
    if eval_hessian:
        h_x = np.zeros((len(x), len(x)))
        return f_x, g_x, h_x
    return f_x, g_x


def expo_function(x: np.ndarray, eval_hessian: bool = False):
    f_x = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g_x = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                    3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    if eval_hessian:
        h_x = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                         3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                        [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                         9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
        return f_x, g_x, h_x
    return f_x, g_x


###################### Constrained functions: ###########################

def eq_constraints_mat(eval_quad: bool = False):
    A = None
    if eval_quad:
        A = np.array([1, 1, 1]).reshape(1, -1)
    return A


def eq_constraints_rhs(eval_quad: bool = False):
    b = None
    if eval_quad:
        b = np.array([1])
    return b


def ineq_const(x: np.ndarray, eval_quad=False):
    # if it is the quadratic function it will return this constraint:
    if eval_quad:
        return [-x[0], -x[1], -x[2]]
    # if it is the linear function will return those:
    else:
        return [-x[0] - x[1] + 1, x[1] - 1, x[0] - 2, -x[1]]


def qp_func(x: np.ndarray, t, eval_hessian: bool = False):
    # x=x0, y=x1, z=x2
    f_x = t * (x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2) - np.log(x[0]) - np.log(x[1]) - np.log(x[2])
    x_deriv = 2 * t * x[0] - 1 / x[0]
    y_deriv = 2 * t * x[1] - 1 / x[1]
    z_deriv = 2 * t * (x[2] + 1) - 1 / x[2]

    g_x = np.array([x_deriv, y_deriv, z_deriv])
    if eval_hessian:
        h_x = np.diag([2 * t + 1 / ((x[0]) ** 2), 2 * t + 1 / ((x[1]) ** 2), 2 * t + 1 / ((x[2]) ** 2)])
        return f_x, g_x, h_x
    return f_x, g_x


def lp_func(x: np.ndarray, t, eval_hessian: bool = False):
    # x=x0, y=x1
    f_x = -t * x[0] - t * x[1] - np.log(x[0] + x[1] - 1) - np.log(1 - x[1]) - np.log(2 - x[0]) - np.log(x[1])
    x_deriv = -t - 1 / (x[0] + x[1] - 1) + 1 / (2 - x[0])
    y_deriv = -t - 1 / (x[0] + x[1] - 1) + 1 / (1 - x[1]) - 1 / x[1]
    g_x = np.array([x_deriv, y_deriv])
    if eval_hessian:
        h_x = np.diag([1 / ((2 - x[0]) ** 2), 1 / ((1 - x[1]) ** 2) + 1 / (x[1] ** 2)])
        h_x += 1 / ((x[0] + x[1] - 1) ** 2)
        return f_x, g_x, h_x
    return f_x, g_x

#
# def qp_constrained_func(x: np.ndarray, eval_hessian: bool = False):
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
# def lp_constrained_func(x: np.ndarray, eval_hessian: bool = False):
#     # x=x0, y=x1
#     f_x = -x[0] - x[1]
#     g_x = np.array([-1, -1])
#     if eval_hessian:
#         h_x = np.zeros((len(x), len(x)))
#         return f_x, g_x, h_x
#     return f_x, g_x
