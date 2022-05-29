import unittest
from src.constrained_min import *
from examples import *
import numpy as np


class TestConstrained(unittest.TestCase):
    func_list = [qp_func, lp_func]  #
    titles_list = ['qp_func', 'lp_func']  # linear_func
    x0_list = [np.array([0.1, 0.2, 0.7]), np.array([0., 0.75])]
    max_iter_list = [1000, 1000]
    eval_quad = [True, False]

    for f, title, x0, max_iter, eval_quad in zip(func_list, titles_list, x0_list, max_iter_list, eval_quad):
        results_list = []
        print(40 * "#", title, 40 * "#")
        ineq = ineq_const(x0, eval_quad)
        A = eq_constraints_mat(eval_quad)
        b = eq_constraints_rhs(eval_quad)
        result = interior_pt(f, ineq, A, b, x0, eval_quad, max_iter)
        results_list.append(result)
        # x1_list = [s[0] for s in results_list]
        # x2_list = [s[1] for s in results_list]
        # object_list = [s[2] for s in results_list]
        # plot_contour_lines(f, methods, title, x1_list, x2_list, max_iter)
        # plot_func_value_vs_iter_num(object_list, methods, title)


if __name__ == '__main__':
    unittest.main()
