import numpy as np
from tests.examples import *


def termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol, g_val=None, h_val=None):
    """
    :return: False - continue to search, True - stop searching.
    """
    diff_param = np.linalg.norm(x_next - x_prev)
    diff_obj = abs(f_prev - f_next)
    if g_val is not None and h_val is not None:
        search_dir = np.linalg.solve(h_val, -g_val)
        nt_decrement_cond = 0.5 * ((search_dir.T.dot(h_val).dot(search_dir)) ** 2)
        return diff_obj < obj_tol or diff_param < param_tol or nt_decrement_cond < obj_tol
    else:
        return diff_obj < obj_tol or diff_param < param_tol


def wolfe_step_len(x, f, method, f_val, g_val, h_val, wolfe_slope_const=0.01, backtrack_const=0.5):
    alpha = 1
    if method.lower() == 'gd':
        pk = -g_val
    elif method.lower() == 'nt':
        pk = np.linalg.solve(h_val, -g_val)
    while f(x + alpha * pk)[0] > f_val + wolfe_slope_const * alpha * np.matmul(g_val.T, pk):
        alpha *= backtrack_const
    return alpha


def minimizer(f, x0, method, step_len, max_iter, obj_tol=1e-12, param_tol=1e-8):
    if method.lower() == 'gd':
        return gd_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter)
    elif method.lower() == 'nt':
        return nt_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter)
    else:
        print("You inserted wrong method please try again")


def gd_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter):
    # Calculate the step len based on wolfe:
    if str(step_len).lower() == 'wolfe':
        f_val, g_val = f(x0)
        alpha = wolfe_step_len(x0, f, 'gd', f_val=f_val, g_val=g_val, h_val=False)
    else:
        alpha = step_len

    x_prev = x0
    f_prev, g_prev = f(x_prev)
    i = 0
    success = False

    print(f"i={i}, x={x_prev}, f(x{i})={f_prev}")

    path_x1_list = [x_prev[0]]
    path_x2_list = [x_prev[1]]
    path_obj_func_list = [f_prev]

    while i < max_iter:
        x_next = x_prev - alpha * g_prev
        f_next, g_next = f(x_next)
        if str(step_len).lower() == 'wolfe':
            alpha = wolfe_step_len(x_next, f, 'gd', f_val=f_next, g_val=g_next, h_val=False)
        else:
            alpha = step_len

        success = termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        if success:
            print(f"Success: {success}")
            return path_x1_list, path_x2_list, path_obj_func_list, success
        else:
            x_prev, f_prev, g_prev = x_next, f_next, g_next

        i += 1
        path_x1_list.append(x_prev[0])
        path_x2_list.append(x_prev[1])
        path_obj_func_list.append(f_prev)
        print(f"i={i}, x={x_prev}, f(x{i})={f_prev}")

    print(f"Success: {success}")
    return path_x1_list, path_x2_list, path_obj_func_list, success


def nt_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter):
    if str(step_len).lower() == 'wolfe':
        f_val, g_val, h_val = f(x0, eval_hessian=True)
        alpha = wolfe_step_len(x0, f, 'nt', f_val=f_val, g_val=g_val, h_val=h_val)
    else:
        alpha = step_len

    x_prev = x0
    f_prev, g_prev, h_prev = f(x0, eval_hessian=True)
    i = 0
    success = False

    print(f"i={i}, x={x_prev}, f(x{i})={f_prev}")

    path_x1_list = [x_prev[0]]
    path_x2_list = [x_prev[1]]
    path_obj_func_list = [f_prev]

    while i < max_iter:
        search_dir = np.linalg.solve(h_prev, -g_prev)
        x_next = x_prev + alpha * search_dir
        f_next, g_next, h_next = f(x_next, eval_hessian=True)
        if str(step_len).lower() == 'wolfe':
            alpha = wolfe_step_len(x_next, f, 'nt', f_val=f_next, g_val=g_next, h_val=h_next)
        else:
            alpha = step_len

        success = termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol, g_val=g_prev, h_val=h_prev)
        if success:
            print(f"Success: {success}")
            return path_x1_list, path_x2_list, path_obj_func_list, success
        else:
            x_prev, f_prev, g_prev, h_prev = x_next, f_next, g_next, h_next

        path_x1_list.append(x_prev[0])
        path_x2_list.append(x_prev[1])
        path_obj_func_list.append(f_prev)
        i += 1
        print(f"i={i}, x={x_prev}, f(x{i})={f_prev}")

    print(f"Success: {success}")
    return path_x1_list, path_x2_list, path_obj_func_list, success
