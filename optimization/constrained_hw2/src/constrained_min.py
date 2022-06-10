from tests.examples import *


def log_bar(f, ineq_const_list, t, x):
    f_x, g_x, h_x = f(x)
    f_x, g_x, h_x = t * f_x, t * g_x, t * h_x
    for ineq in ineq_const_list:
        f_i, g_i, h_i = ineq(x)
        f_x = f_x - np.log(-f_i)
        g_x = g_x - (1 / f_i) * g_i
        h_x = h_x + (1 / f_i ** 2) * g_i @ g_i.T - (1 / f_i) * h_i
    return f_x, g_x, h_x


def calc_search_dir_equality_contst(g_x, h_x, A):
    temp1 = np.vstack([np.hstack([h_x, A.T]),
                       np.hstack([A, np.zeros((1, A.shape[0]))])])
    temp2 = np.vstack((-g_x, np.zeros((1, A.shape[0]))))
    result = np.linalg.solve(temp1, temp2)
    return result[:A.shape[1]].reshape((-1)), result[A.shape[1]:].reshape((-1))


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


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, max_inner_iter=100, mu=10,
                epsilon=1e-6):
    x_prev = x0.copy()
    t = 1  # V
    # number of inequality constraint:
    m = len(ineq_constraints)
    # Check if there are equality constraints:
    A, b = eq_constraints_mat, eq_constraints_rhs

    # constants variables:
    # wolfe_slope_const=0.5
    # backtrack_const = 0.9
    wolfe_slope_const = 0.01
    backtrack_const = 0.5
    obj_tol = 1e-12
    param_tol = 1e-8

    if A is None:
        method = 'nt'
    else:
        method = 'nt_equality'

    path_history = []
    i = 1

    # Appending the inital point:
    w_k = np.zeros((1))
    func_val, _, _ = func(x_prev)
    path_history.append(list(x_prev) + [func_val, i] + list(w_k / t))

    # Outer loop:
    while m / t > epsilon:

        print(f"iteration: #{i}, x value:{x_prev}")
        if i != 1:
            t *= mu
        i += 1
        f_prev, g_prev, h_prev = log_bar(func, ineq_constraints, t, x_prev)
        j = 0
        success = False

        # Inner loop:
        while not success and j < max_inner_iter:
            alpha = 1
            if method == 'nt_equality':
                search_dir, w_k = calc_search_dir_equality_contst(g_prev, h_prev, A)
            else:

                search_dir = -np.linalg.solve(h_prev, np.identity(len(h_prev))) @ g_prev
                search_dir = search_dir.reshape((-1))
                w_k = np.zeros((1))

            f_next, g_next, h_next = log_bar(func, ineq_constraints, t, x_prev + alpha * search_dir)

            if np.isnan(f_next):
                f_next = np.inf

            # calculate alpha based on wolfe step:
            while f_next > f_prev + wolfe_slope_const * alpha * (g_prev.T @ search_dir):
                alpha *= backtrack_const
                f_next, g_next, h_next = log_bar(func, ineq_constraints, t, x_prev + alpha * search_dir)
                if np.isnan(f_next):
                    f_next = np.inf

            x_next = x_prev + alpha * search_dir

            f_next, g_next, h_next = log_bar(func, ineq_constraints, t, x_next)

            success = termination_flag(x_next, x_prev, f_next, f_prev, obj_tol, param_tol, g_val=g_prev, h_val=h_prev)

            if success:
                func_val, g_val, h_val= func(x_next)
                path_history.append(list(x_next) + [func_val, i] + list(w_k / t))
            else:
                j += 1
            x_prev, f_prev, g_prev, h_prev = x_next, f_next, g_next, h_next

    path_history = np.array(path_history)
    return x_prev, path_history # x_prev, path_history
