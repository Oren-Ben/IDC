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
        h_x = np.array([[-400.0 * x[1] + 1200 * x[0] ** 2 + 2,-400 * x[0]], # Check if it is 2-2 or 2+2
                         [-400 * x[0], 200]])
        return f_x, g_x, h_x
    return f_x, g_x


def linear_func(x: np.ndarray, eval_hessian: bool = False ):
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
