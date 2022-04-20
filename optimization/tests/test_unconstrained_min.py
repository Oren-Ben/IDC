import unittest
from src.unconstrained_min import *
from examples import *
from src.utils import *
import numpy as np

class TestUnconstrained(unittest.TestCase):

    func_list = [f_calc_d1, f_calc_d2, f_calc_d3, expo_function, rosenbrock_func] # linear_func
    titles_list = ['Quadratic_1','Quadratic_2','Quadratic_3', 'Sum of exponents', 'Rosenbrock'] # linear_func
    x0_list = [np.array([1.,1.]),np.array([1.,1.]), np.array([1.,1.]),np.array([1.,1.]), np.array([-1.,2.])]
    max_iter_list = [100, 100, 100, 100, 10000]

    for f, title, x0, max_iter in zip(func_list,titles_list,x0_list,max_iter_list):
        results_list=[]
        methods=['gd','nt']
        print(40*"#",title,40*"#")
        for method in methods:
            print(15*"#", method.upper(),"Method",15*"#")
            result = minimizer(f=f, x0=x0,method=method,step_len='wolfe',max_iter=max_iter)
            results_list.append(result)
        x1_list = [s[0] for s in results_list]
        x2_list = [s[1] for s in results_list]
        object_list = [s[2] for s in results_list]
        plot_func_value_vs_iter_num(object_list, methods, title)

    #print(np.arange(len(object_list[0])))


#     def test_f1(self):
#         x0_rosenbrock = np.array([-1, 2]).T
#         result = src.unconstrained_min.minimizer(rosenbrock_func, x0_rosenbrock,'GD', 'Wolfe', 1e-12, 1e-8, 100)
#
unittest.main()