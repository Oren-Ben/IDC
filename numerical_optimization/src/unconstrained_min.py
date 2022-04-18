import numpy as np
import math

def termination_flag(current_diff_obj:float, obj_tol:float, current_diff_param:float, param_tol:float):
    """
    :param current_diff_obj: the difference between the previous and current objective function values
    :param obj_tol: the numeric tolerance for successful termination in terms of f(x)
    :param current_diff_param: the difference between the previous and current x values
    :param param_tol: the numeric tolerance for successful termination in terms of x
    :return: False - continue to search, True - stop searching.
    """
    return current_diff_obj<=obj_tol or current_diff_param<=param_tol

def minimization_func(f, x0, step_len, obj_tol, param_tol, max_iter):
    if type(step_len) == str:
        if step_len.lower() == 'wolfe':
                step_len = 1e-8
        else:
            print("Typo mistake, please try again")
            return -1
        

    x_values = []
    f_values = []


    # else:
    #     step_len = 0.01
    #
    # return step_len
    # return final_loc, final_obj_val, success_flag


x = np.array((1, 1))
f = x[0] + x[1]
print(minimization_func(f, x, 'Wolfe', 1e-12, 1e-8, 100))
