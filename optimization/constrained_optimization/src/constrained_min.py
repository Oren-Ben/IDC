import numpy as np

from src.unconstrained_min import *
from tests.examples import  *



def interior_pt(func,ineq_constraints,eq_constraints_mat,eq_constraints_rhs,x0
                ,eval_quad, max_inner_loops, mu = 10, epsilon = 1e-6):

    x_hist = None
    x_new = x0.copy()
    success = False
    max_outer_iter = 100
    t =1
    # number of inequality constraint:
    # func is the class (QPFunction/LPFunction)
    m = len(ineq_constraints)

    # Check if there are equality constraints:
    A , b = eq_constraints_mat, eq_constraints_rhs
    if A is None:
        method = 'nt'
    else:
        method = 'nt_equality'

    for i in range(max_outer_iter):
        _, x_temp, _, path_x_hist_temp = minimizer(func,x_new,method,'wolfe',max_inner_loops,t,A)

        if x_hist is None:
            x_hist = path_x_hist_temp
        else:
            x_hist = np.append(x_hist, path_x_hist_temp, axis=1)

        if m/t <epsilon:
            success = True
            break

        x_new = x_temp
        t *= mu

    path_obj_func_list = [func(x)[0] for x in x_hist.T]
    return success, x_new, path_obj_func_list, x_hist
