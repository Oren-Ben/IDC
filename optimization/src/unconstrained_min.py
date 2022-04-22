import numpy as np
from tests.examples import *


# Comments April 20th:

# Linear function - need to check, currently doesn't work in GD. Need to check in NT.
# Problem - Rosenbrock with GD and regular step size -
# doesn't decrease the function value, but it does stop after the first iteration when the func value increase.

# Gradient for all the qudartic functions, expo (both wolfe and regular step) - WORK!
# Newton for all the qudartic functions, expo (both wolfe and regular step) - WORK!
# Rosenbrock -
# GD & wolfe - only 1 step that has the same function value (804)
# GD & regular step - increase the function value with one step from 804 to 2019 and then stops.
# NT (both wolfe and regular step) - looks good.
# linear function - works for GD, need to work only for GD.

# TODO - Linear func with NT - need to add try and except for the nigular matrix. (the second derivitive is 0)


# def inverse_calculation(matrix):
#     return np.linalg.solve(matrix, np.eye(matrix.shape[0]))

def termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol, g_val=None, h_val=None):
    """
    :return: False - continue to search, True - stop searching.
    """
    diff_param = sum(abs(x_next - x_prev))
    diff_obj = abs(f_prev - f_next)
    if g_val is not None and h_val is not None:
        search_dir = np.linalg.solve(h_val, -g_val)
        nt_decrement_cond = 0.5*((search_dir.T.dot(h_val).dot(search_dir))**2)
        print('diff_param')
        print(diff_param)
        print('diff_obj')
        print(diff_obj)
        print('nt_decrement_cond')
        print(nt_decrement_cond)
        return diff_obj < obj_tol or diff_param < param_tol or nt_decrement_cond<obj_tol
    else:
        print('diff_param')
        print(diff_param)
        print('diff_obj')
        print(diff_obj)
        return diff_obj < obj_tol or diff_param < param_tol


def wolfe_step_len(x, f, method, f_val, g_val, h_val, wolfe_slope_const=0.01, backtrack_const=0.5):
    step_len = 1
    if method.lower() == 'gd':
        pk = -g_val
    elif method.lower() == 'nt':
        # need to change the implementation of the linalg.solve:
        pk = np.linalg.solve(h_val, -g_val)
    while f(x + step_len * pk)[0] > f_val + wolfe_slope_const * g_val.T.dot(pk) * step_len:
        step_len *= backtrack_const
    return step_len


# adding to the minimizer function the parameter "method" in according to Yonathan response on piazza.
def gd_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter):
    # Calculate the step len based on wolfe:
    if type(step_len) == str:
        if step_len.lower() == 'wolfe':
                f_val, g_val = f(x0)
                step_len = wolfe_step_len(x0, f, 'gd', f_val=f_val, g_val=g_val, h_val=False)
        else:
            print("Not wolfe")

    x_prev = x0
    f_prev, g_prev = f(x_prev)
    i = 0
    success = False

    print(f"i={i}, x={x_prev}, f(x{i})={f_prev}") #i, x_prev, f_prev, success

    path_x1_list = [x_prev[0]]
    path_x2_list = [x_prev[1]]
    path_obj_func_list = [f_prev]

    # x - the points, will change in each iteration until reaches the flag termination.
    # f(x)[1] - will have the gradienet
    # Check if the termination rule exist?
    # report the postion: i, x[i], f[x[i]] (We should save only x[i] and f[x[i]])
    # at the end final location and success/failure.
    while not success and i < max_iter:
        x_next = x_prev - step_len * g_prev
        f_next, g_next = f(x_next)
        i += 1
        path_x1_list.append(x_next[0])
        path_x2_list.append(x_next[1])
        path_obj_func_list.append(f_next)
        success = termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        print(f"i={i}, x={x_next}, f(x{i})={f_next}")
        if not success:
            x_prev, f_prev, g_prev = x_next, f_next, g_next
    # Think in the future if we should add it in the return: iteration_report_dict[i],
    print(f"Succes: {success}")
    return path_x1_list, path_x2_list, path_obj_func_list, success


def nt_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter):

    if type(step_len) == str:
        if step_len.lower() == 'wolfe':
                f_val, g_val, h_val = f(x0,eval_hessian=True)
                step_len = wolfe_step_len(x0, f, 'nt', f_val=f_val, g_val=g_val, h_val=h_val)
        else:
            print("Not wolfe")

    x_prev = x0
    f_prev, g_prev, h_prev = f(x0, eval_hessian=True)
    i = 0
    success = False

    print(f"i={i}, x={x_prev}, f(x{i})={f_prev}")

    path_x1_list = [x_prev[0]]
    path_x2_list = [x_prev[1]]
    path_obj_func_list = [f_prev]

    while not success and i < max_iter:
        # search_dir = -inverse_calculation(h_prev).dot(g_prev)
        search_dir = np.linalg.solve(h_prev, -g_prev)
        x_next = x_prev + step_len * search_dir
        f_next, g_next, h_next = f(x_next, eval_hessian=True)
        i += 1
        path_x1_list.append(x_next[0])
        path_x2_list.append(x_next[1])
        path_obj_func_list.append(f_next)
        success = termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol,g_val=g_prev, h_val=h_prev)
        print(f"i={i}, x={x_next}, f(x{i})={f_next}")
        if not success:
            x_prev, f_prev, g_prev, h_prev = x_next, f_next, g_next, h_next
    print(f"Succes: {success}")
    return path_x1_list, path_x2_list, path_obj_func_list, success



def minimizer(f, x0, method, step_len, max_iter, obj_tol=1e-12, param_tol=1e-8):

    if method.lower() =='gd':
        return gd_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter)
    elif method.lower() == 'nt':
        return nt_minimizer(f, x0, step_len, obj_tol, param_tol, max_iter)
    else:
        print("You inserted wrong method please try again")


if __name__ == '__main__':
    x0 = np.array([1, 1]).T
    x0_rosenbrock = np.array([-1, 2]).T
    path_x1_list, path_x2_list, path_obj_func_list, success = minimizer(f_calc_d1, x0, 'gd', 0.01, 100)

