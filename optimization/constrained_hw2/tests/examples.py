import numpy as np


# equality constraints, general functions for both QP and LP:
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

###################### QP function: ###########################

def qp_func(x: np.ndarray):  # , eval_quad: bool = False):
    # x=x0, y=x1, z=x2
    # min function
    f_x = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    g_x = np.array([[2 * x[0],
                     2 * x[1],
                     2 * x[2] + 2]]).T
    h_x = np.array([[2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2]])
    return f_x, g_x, h_x


# QP inequality constraints addition:

def qp_ineq1(x: np.ndarray):
    f_x = -x[0]
    g_x = np.array([[-1, 0, 0]]).T
    h_x = np.zeros((3, 3))

    return f_x, g_x, h_x


def qp_ineq2(x: np.ndarray):
    f_x = -x[1]
    g_x = np.array([[0, -1, 0]]).T
    h_x = np.zeros((3, 3))

    return f_x, g_x, h_x


def qp_ineq3(x: np.ndarray):
    f_x = -x[2]
    g_x = np.array([[0, 0, -1]]).T
    h_x = np.zeros((3, 3))

    return f_x, g_x, h_x


###################### LP function: ###########################

def lp_func(x: np.ndarray):  # , eval_quad: bool = False):
    # x=x0, y=x1
    # adding minus because its a maximize function
    f_x = -x[0] - x[1]
    g_x = np.array([[-1, -1]]).T
    h_x = np.zeros((len(x), len(x)))
    return f_x, g_x, h_x


def lp_ineq1(x: np.ndarray):
    f_x = -x[0] - x[1] + 1
    g_x = np.array([[-1, -1]]).T
    h_x = np.zeros((2, 2))

    return f_x, g_x, h_x


def lp_ineq2(x: np.ndarray):
    f_x = x[1] - 1
    g_x = np.array([[0, 1]]).T
    h_x = np.zeros((2, 2))

    return f_x, g_x, h_x


def lp_ineq3(x: np.ndarray):
    f_x = x[0] - 2
    g_x = np.array([[1, 0]]).T
    h_x = np.zeros((2, 2))

    return f_x, g_x, h_x


def lp_ineq4(x: np.ndarray):
    f_x = -x[1]
    g_x = np.array([[0, -1]]).T
    h_x = np.zeros((2, 2))

    return f_x, g_x, h_x
