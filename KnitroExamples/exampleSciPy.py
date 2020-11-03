#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Artelys Knitro with scipy package
# to solve different types of optimization problems directly within scipy.
#
# A detailed description of the data structure available to pass
# information to set up Knitro in scipy is presented in 
# knitro\scipy\scipy_wrapper.py. A more generic documentation is
# available in the scipy.optimize.minimize documentation.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
from scipy.optimize import minimize,OptimizeResult, Bounds, LinearConstraint, \
                           NonlinearConstraint, BFGS, SR1, show_options
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import LinearOperator

from knitro import *
from knitro.numpy import *
from knitro.scipy import kn_minimize

### Knitro-Scipy LP1 example
def example_scipy_LP1():
    # A simple Linear Problem (LP)
    # minimize     -4*x0 - 2*x1
    #     subject to   x0 + x1 + x2        = 5
    #                  2*x0 + 0.5*x1 + x3  = 8
    #                 0 <= (x0, x1, x2, x3)
    #  The optimal solution is:
    #     obj=17.333 x=[3.667,1.333,0,0]

    # Define the objective
    def objective(x):
        x = np.asarray(x)
        return -4 * x[0] - 2 * x[1]

    # Define the objective gradient
    def objGradient(x):
        x = np.asarray(x)
        der = np.zeros_like(x)
        der[0] = -4
        der[1] = -2
        der[2] = 0
        return der

    # Define variables
    ## initial point
    x0 = [10, 10, 10, 10]
    
    ## bounds
    bounds = Bounds([0, 0, 0, 0],
                    [KN_INFINITY, KN_INFINITY, KN_INFINITY, KN_INFINITY])

    # Define linear constraints
    linear_constraint = LinearConstraint([[1, 1, 1, 0], [2, 0.5, 0, 1]],
                                         [5, 8], [5, 8])

    # Solve the problem using scipy minimize function
    # Simply replace the 'method' by kn_minimize
    result = minimize(
        fun=objective,
        x0=x0,
        method=kn_minimize,
        constraints=[linear_constraint],
        bounds=bounds,
        jac=objGradient)


### Knitro-Scipy NLP1 example
def example_scipy_NLP1(sparsity_pattern=False):
    # min   100 (x1 - x0^2)^2 + (1 - x0)^2
    # s.t.  x0 x1 >= 1
    #       x0 + x1^2 >= 0
    #       x0 <= 0.5
    #
    # The standard start point (-2, 1) usually converges to the standard
    # minimum at (0.5, 2.0), with final objective = 306.5.
    # Sometimes the solver converges to another local minimum
    # at (-0.79212, -1.26243), with final objective = 360.4.
    # This problem also presents how to pass the objective and
    # constraints sparsity pattern.

    # Definition of the objective function
    def fun(x):
        x = np.asarray(x)
        dTmp = x[1] - x[0] * x[0]
        return 100.0 * (dTmp * dTmp) + ((1.0 - x[0]) * (1.0 - x[0]))

    # Definition of the dense objective gradient
    def fun_der(x):
        x = np.asarray(x)
        der = np.zeros_like(x)

        dTmp = x[1] - x[0] * x[0]
        der[0] = (-400.0 * dTmp * x[0]) - (2.0 * (1.0 - x[0]))
        der[1] = 200.0 * dTmp
        return der

    # Definition of the dense Hessian function for the objective
    def fun_hess(x):
        x = np.asarray(x)
        H = np.zeros((x.shape[0], x.shape[0]))

        H[0, 0] = (-400.0 * x[1]) + (1200.0 * x[0] * x[0]) + 2.0  # (0,0)
        H[0, 1] = -400.0 * x[0]  # (0,1)
        H[1, 1] = 200.0  # (1,1)
        return H

    # Definition of the sparse objective gradient
    def fun_der_sparse(x):
        x = np.asarray(x)
        der = np.zeros_like(x)

        dTmp = x[1] - x[0] * x[0]
        der[0] = (-400.0 * dTmp * x[0]) - (2.0 * (1.0 - x[0]))
        der[1] = 200.0 * dTmp

        der = csc_matrix(der)
        return der

    # Definition of the sparse Hessian function for the objective
    def fun_hess_sparse(x):
        x = np.asarray(x)
        H = np.zeros((x.shape[0], x.shape[0]))

        H[0, 0] = (-400.0 * x[1]) + (1200.0 * x[0] * x[0]) + 2.0  # (0,0)
        H[0, 1] = -400.0 * x[0]  # (0,1)
        H[1, 1] = 200.0  # (1,1)

        H = csc_matrix(H)
        return H

    # Initial point
    x0 = [-2.0, 1.0]
    
    # Variables bounds
    bounds = Bounds([-np.inf, -np.inf], [0.5, np.inf])

    # Definition of the nonlinear constraints
    #       x0 x1 >= 1
    #       x0 + x1^2 >= 0
    def cons_f(x):
        return [x[0] * x[1], x[0] + x[1]**2]

    # Nonlinear constraint Jacobian
    def cons_J(x):
        return [[x[1], x[0]], [1, 2 * x[1]]]

    # Nonlinear constraint Hessian
    def cons_H(x, v):
        return v[0] * np.array([[0, 1], [1, 0]]) \
               + v[1] * np.array([[0, 0], [0, 2]])

    # Sparse nonlinear constraint Jacobian
    def cons_J_sparse(x):
        return csc_matrix([[x[1], x[0]], [1, 2 * x[1]]])

    # Sparse nonlinear constraint Hessian
    def cons_H_sparse(x, v):
        return csc_matrix(v[0] * np.array([[0, 1], [1, 0]]) \
               + v[1] * np.array([[0, 0], [0, 2]]))

    # Constraints bounds
    cL = [1, 0]
    cU = [np.inf, np.inf]

    if sparsity_pattern:
        # Objective sparsity pattern
        objGradIndexVars = [0, 1]
        H = np.zeros((len(x0), len(x0)))
        H[0, 0] = 1
        H[0, 1] = 1
        H[1, 1] = 1
        sparsity_hess_obj = csc_matrix(H)
        hessIndexVars1_obj = sparsity_hess_obj.nonzero()[0]
        hessIndexVars2_obj = sparsity_hess_obj.nonzero()[1]

        # Constraints sparsity pattern
        sparsity_jac = [[1, 1], [1, 2]]
        sparsity_jac = csc_matrix(sparsity_jac)
        jacIndexCons = sparsity_jac.nonzero()[0]
        jacIndexVars = sparsity_jac.nonzero()[1]
        sparsity_hess = [[0, 1], [0, 2]]
        sparsity_hess = csc_matrix(sparsity_hess)
        hessIndexVars1 = sparsity_hess.nonzero()[0]
        hessIndexVars2 = sparsity_hess.nonzero()[1]
        
        # Define the constraints sparsity pattern 
        # jacIndexCons: Store nnzj indexes (row) of each nonzero
        # in the Jacobian of the constraints.
        # jacIndexVars: Store nnzj index (column) of each nonzero
        # in the Jacobian of the constraints.
        # hessIndexVars1: Store nnzh index of each nonzero
        # in the Hessian of the Lagrangian.
        # hessIndexVars2: Store nnzh index of each nonzero
        # in the Hessian of the Lagrangian.
        finite_diff_jac_sparsity = {'jacIndexCons': jacIndexCons,
                                    'jacIndexVars': jacIndexVars,
                                    'hessIndexVars1': hessIndexVars1,
                                    'hessIndexVars2': hessIndexVars2}

        # Define the nonlinear constraints with the sparsity pattern
        nonlinear_constraint = NonlinearConstraint(
            cons_f, cL, cU, jac=cons_J_sparse, hess=cons_H_sparse,
            finite_diff_jac_sparsity= finite_diff_jac_sparsity)

        # Solve the problem using the sparsity pattern
        # The objective derivatives sparsitty pattern is directly passed
        # to the scipy minimize function
        res = minimize(
            fun=fun,
            x0=x0,
            method=kn_minimize,
            jac=fun_der_sparse,
            hess=fun_hess_sparse,
            constraints=[nonlinear_constraint],
            bounds=bounds,
            options={'outlev': 0,
                     'sparse':{'objGradIndexVars': objGradIndexVars,
                               'hessIndexVars1': hessIndexVars1_obj,
                               'hessIndexVars2': hessIndexVars2_obj }})
    
    else:
        # Define the nonlinear constraints without sparsity pattern
        nonlinear_constraint = NonlinearConstraint(cons_f, cL, cU, jac=cons_J, hess=cons_H)
        
        # Add options
        options = {}
        options['disp'] = True

        # Solve the problem without the sparsity pattern
        res = minimize(
            fun=fun,
            x0=x0,
            method=kn_minimize,
            jac=fun_der,
            hess=fun_hess,
            constraints=[nonlinear_constraint],
            bounds=bounds,
            options=options)


### Knitro-Scipy NLP2 example
def example_scipy_NLP2():
    # max   x0*x1*x2*x3         (obj)
    # s.t.  x0^3 + x1^2 = 1     (c0)
    #       x0^2*x3 - x2 = 0    (c1)
    #       x3^2 - x1 = 0       (c2)
    # This example also shows to use the callback object in
    # scipy.optimize.minimize, which can be used to perform some user-defined
    # task every time Knitro iterates to a new solution estimate
    # (e.g. it can be used to define a customized stopping condition).

    # Definition of the objective function
    def objective(x):
        return x[0] * x[1] * x[2] * x[3]

    # Definition of the objective gradient
    def objGrad(x):
        x = np.asarray(x)
        der = np.zeros_like(x)
        der[0] = x[1] * x[2] * x[3]
        der[1] = x[0] * x[2] * x[3]
        der[2] = x[0] * x[1] * x[3]
        der[3] = x[0] * x[1] * x[2]
        return der

    # Definition of the objective Hessian
    def objHess(x):
        x = np.asarray(x)
        H = np.zeros((x.shape[0], x.shape[0]))
        #Note: We only need to give the upper triangular matrix.
        #It works with the full matrix as well.
        H[0, 1] = x[2] * x[3]
        H[0, 2] = x[1] * x[3]
        H[0, 3] = x[1] * x[2]
        H[1, 2] = x[0] * x[3]
        H[1, 3] = x[2] * x[3]
        H[2, 3] = x[0] * x[1]
        return H

    # Definition of the  variables
    # Definition of the initial point
    x0 = [0.8, 0.8, 0.8, 0.8]

    # Definition of the constraints
    #       x0^3 + x1^2 = 1     (c0)
    #       x0^2*x3 - x2 = 0    (c1)
    #       x3^2 - x1 = 0       (c2)
    def cons_f(x):
        return [
            x[0] * x[0] * x[0] + x[1] * x[1],
            x[0] * x[0] * x[3] - x[2],
            x[3] * x[3] - x[1]
               ]

    # Nonlinear constraint Jacobian
    def cons_J(x):
        jac_cons1 = [3 * x[0]**2, 2 * x[1], 0, 0]
        jac_cons2 = [2 * x[0] * x[1], 0, -1, x[0] * x[0]]
        jac_cons3 = [0, -1, 0, 2 * x[3]]
        return [jac_cons1, jac_cons2, jac_cons3]

    # Nonlinear constraint Hessian
    def cons_H(x, v):
        hess_cons1 = np.array([[6 * x[0], 0, 0, 0], [0, 2, 0, 0, ],
            [0, 0, 0, 0, ], [0, 0, 0, 0]])
        hess_cons2 = np.array([[2 * x[3], 0, 0, 2 * x[0]],
            [0, 0, 0, 0, ], [ 0, 0, 0, 0], [2 * x[0], 0, 0, 0]])
        hess_cons3 = np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                               [0, 0, 0, 0], [0, 0, 0, 2]])
        return v[0] * hess_cons1 + v[1] * hess_cons2 + v[2] * hess_cons3

    # Constraints lower and upper bounds
    cL = [1, 0, 0]
    cU = [1, 0, 0]

    bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf])

    nonlinear_constraint = NonlinearConstraint(
        cons_f, cL, cU, jac=cons_J, hess=cons_H)

    # Example of external callback. The callback function passed here is
    # invoked after Knitro computes a new estimate of the solution.
    def callbackF(x):
        print(">> New point computed by Knitro: (",
              ",".join("%20.12e" % xi for xi in x), ")")
        print(">> New objective computed by Knitro: (", objective(x), ")")

    # Solve the problem using the scipy minimize function
    res = minimize(
        fun=objective,
        x0=x0,
        method=kn_minimize,
        jac=objGrad,
        hess=objHess,
        constraints=[nonlinear_constraint],
        bounds=bounds,
        callback=callbackF)


### Knitro-Scipy Rosenbrock example
def example_scipy_rosenbrock(is_sparse=False):
    #   min 100*(x2-x1*x1)^2+(1-x1)^2   (obj)
    #   st  x0 + 2*x1 <= 1
    #       x0**2 + x1 <= 1
    #       x0**2 -x1 <= 1
    #       2*x0 + x1 = 1
    #       0 <= x0 <= 1
    #       -0.5 <= x1 <= 2.0
    # Initial point x0 = (0.5, 0)
    # This example also shows how to use the sparse matrix, which
    # can be used to improve Knitro efficiency.

    # Definition of the objective function
    def rosen(x):
        """The Rosenbrock function"""
        x = np.asarray(x)
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    # Definition of the objective gradient
    def rosen_grad(x):
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        grad = np.zeros_like(x)
        grad[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm \
                    - 2 * (1 - xm)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        grad[-1] = 200 * (x[-1] - x[-2]**2)
        return grad

    # Definition of the objective Hessian
    def rosen_hess(x):
        x = np.asarray(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros_like(x)
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H

    # Definition of the variables
    # Initial point
    x0 = [0.5, 0]
    
    # Variables bounds
    bounds = Bounds([0, -0.5], [1.0, 2.0])

    # Definition of the constraints
    # Linear constraints
    linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1],
                                         [1, 1])
    # Nonlinear constraints
    def cons_f(x):
        return [x[0]**2 + x[1], x[0]**2 - x[1]]

    # Provide a sparse structure for Jacobian and Hessian
    if is_sparse:
        # Sparse nonlinear constraints Jacobian
        def cons_J_sparse(x):
            return csc_matrix([[2 * x[0], 1], [2 * x[0], -1]])
    
        # Sparse nonlinear constraint Hessian
        def cons_H_sparse(x, v):
            return v[0] * csc_matrix([[2, 0], [0, 0]]) + \
               v[1] * csc_matrix([[2, 0], [0, 0]])
               
        # Definition of sparse nonlinear constraints
        nonlinear_constraint = NonlinearConstraint(
            cons_f, -np.inf, 1, jac=cons_J_sparse, hess=cons_H_sparse)
    else:
        # Nonlinear constraints Jacobian
        def cons_J(x):
            return [[2 * x[0], 1], [2 * x[0], -1]]
        
        # Nonlinear constraint Hessian
        def cons_H(x, v):
            return v[0] * np.array([[2, 0], [0, 0]]) \
                + v[1] * np.array([[2, 0], [0, 0]])
                
        # Definition of dense nonlinear constraints
        nonlinear_constraint = NonlinearConstraint(
            cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

    # Solve the problem using the scipy minimize function
    res = minimize(
        fun=rosen,
        x0=x0,
        method=kn_minimize,
        jac=rosen_grad,
        hess=rosen_hess,
        constraints=[linear_constraint, nonlinear_constraint],
        bounds=bounds,
        options={'outlev': 6})

example_scipy_LP1()
example_scipy_NLP2()
example_scipy_NLP1()
example_scipy_NLP1(sparsity_pattern=True)
example_scipy_rosenbrock(is_sparse=True)
example_scipy_rosenbrock(is_sparse=False)