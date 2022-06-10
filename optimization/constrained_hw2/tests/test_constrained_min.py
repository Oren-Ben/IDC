import unittest
from src.utils import *

class TestConstrained(unittest.TestCase):
    func_list = [qp_func, lp_func]
    titles_list = ['qp_func', 'lp_func']
    x0_list = [np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.75])]
    eval_quad = [True, False]
    ineq_qp = [qp_ineq1, qp_ineq2, qp_ineq3]
    ineq_lp = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
    ineq = [ineq_qp, ineq_lp]

    for f, title, x0, ineq, eval_quad in zip(func_list, titles_list, x0_list, ineq, eval_quad):
        results_list = []
        print(40 * "#", title, 40 * "#")
        ineq = ineq
        A = eq_constraints_mat(eval_quad)
        b = eq_constraints_rhs(eval_quad)
        x_final, path_history = interior_pt(f, ineq, A, b, x0)

        # Report:
        print("-----")
        print(f"Final Results for the {title}:")
        f_val, _, _ = f(x_final)
        if title == 'lp_func':
            f_val = f_val * (-1)
        print(f"Objective value for the function is: {f_val} and the X: {x_final}")

        print("in-equality contraints:")
        for i, ineq_const in enumerate(ineq):
            if title == 'qp_func':
                ineq_final = ineq_const(x_final)[0] * (-1)
            else:
                if i + 1 == 4:
                    ineq_final = ineq_const(x_final)[0] * (-1)
                else:
                    ineq_final = ineq_const(x_final)[0]
            print(f"#{i + 1} contraint: {ineq_final}")

        # Plot the path
        if title == 'qp_func':
            qp_plot_3d(x0, path_history,title)
        else:
            lp_plot_2d(x0, path_history,title)
        # plot function value vs iteration number:
        plot_func_value_vs_iter_num(x0, path_history, title)


if __name__ == '__main__':
    unittest.main()
