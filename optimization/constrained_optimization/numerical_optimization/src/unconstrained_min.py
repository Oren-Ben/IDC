import numpy as np
from tests.examples import *

def termination_flag(x_next, x_prev, f_next,f_prev, obj_tol, param_tol):
    """
    :return: False - continue to search, True - stop searching.
    """
    diff_obj = np.linalg.norm(x_prev - x_next)
    diff_param = f_prev - f_next
    return diff_obj<=obj_tol or diff_param<=param_tol

def wolfe_step_len(x,f,method,f_val,g_val, h_val,wolfe_slope_const=0.15,backtrack_const=0.1):
    init_step = 1
    if method =='GD':
        pk = -g_val
    elif method == 'NT':
        # need to change the implementation of the linalg.solve:
        pk = np.linalg.solve(h_val,-g_val)
    while f(x+init_step*pk)[0]>f_val+wolfe_slope_const*g_val.T.dot(pk)*init_step:
        init_step*=backtrack_const
    return init_step

# adding to the minimizer function the parameter "method" in according to Yonathan response on piazza.
def minimizer(f, x0,method, step_len, obj_tol, param_tol, max_iter):

    if type(step_len) == str:
        if step_len.lower() == 'wolfe':
            f_val, g_val,h_val = f(x0,eval_hessian=True)
            step_len = wolfe_step_len(x0,f,method,f_val=f_val,g_val=g_val,h_val=h_val)

    # our code
    if method == 'GD':
        # x - the points, will change in each iteration until reaches the flag termination.
        # f(x)[1] - will have the gradienet
        # Check if the termination rule exist?
        # report the postion: i, x[i], f[x[i]] (We should save only x[i] and f[x[i]])
        # at the end final location and success/failure.
        x_prev = x0
        f_prev, g_prev = f(x0)
        i = 0
        print(i, x_prev, f_prev)
        success = False
        iteration_report_dict = {i:[x_prev,f_prev]}
        while not success and i<=max_iter:
            x_next = x_prev - step_len*g_prev
            f_next, g_next = f(x_next)
            i+=1
            iteration_report_dict[i]=[x_next,f_next]
            print(i,x_next,f_next)
            success = termination_flag(x_next,x_prev,f_next,f_prev,obj_tol,param_tol)
            if not success:
                x_prev, f_prev, g_prev = x_next, f_next, g_next
        # Think in the future if we should add it in the return: iteration_report_dict[i],
        return iteration_report_dict, success

    elif method == 'NT':
        print("not ready yet")
        return 1, 2

    else:
        print("You inserted wrong number please try again")

    #
    # if type(step_len) == str:
    #     if step_len.lower() == 'wolfe':
    #             step_len = 1e-8
    #     else:
    #         print("Typo mistake, please try again")
    #         return -1
    #

    # x_values = []
    # f_values = []


    # else:
    #     step_len = 0.01
    #
    # return step_len
    # return final_loc, final_obj_val, success_flag


#x = np.array((1, 1))
#f = x[0] + x[1]
x = np.array([1,1]).T
#print(x.shape)
gd_dict, succ = minimizer(f_calc_d3, x,'NT', 'wolfe', 1e-12, 1e-8, 100)
