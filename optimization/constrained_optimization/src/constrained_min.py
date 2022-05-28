

def interior_pt(func,ineq_constraints,eq_constraints_mat,eq_constraints_rhs,x0):

    x_hist = None
    x_new = x0.copy()
    success = False
    max_outer_iter = 100
    t =1
    # number of inequality constraint:
    # func is the class (QPFunction/LPFunction)
    m = len(func.ineq_const())

    if func.A is None:
        method = 'nt'
    else:
        method = 'nt_equality'

    for i in range(max_outer_iter):
        # update the t?
        # Call the functin from unconstrained_min
    # add to this function the option nt_qeuality
    # when we call the function we want to give it the method and the return should be last x and x hist
    # append the x hist into x_hist param
    # check if the condition m/t < epislon exist, if yes stop if not continue
    # update x
    # update t with mu




    pass


