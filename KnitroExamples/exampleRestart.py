#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Knitro to solve the following
# simple nonlinear optimization problem, while varying user
# options and bounds on variables or constraints.
# This model is test problem HS15 from the Hock & Schittkowski
# collection. 
#
# min   100 (x1 - x0^2)^2 + (1 - x0)^2
# s.t.  x0 x1 >= c0lb (initially 1)
#       x0 + x1^2 >= 0
#       x0 <= x0ub    (initially 0.5)
#
# We first solve the model with c0lb=1 and x0ub=0.5, and then
# re-solve for different values of the "bar_murule" user option.
# We then re-solve the model, while changing the value of the 
# variable bound "x0ub".  Finally, we re-solve while changing the 
# value of the constraint bound "c0lb".
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

#*------------------------------------------------------------------* 
#*     FUNCTION callbackEvalF                                       *
#*------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "obj" is set in the KN_eval_result structure.
def callbackEvalF (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear objective
    dTmp = x[1] - x[0]*x[0]
    evalResult.obj = 100.0 * (dTmp*dTmp) + ((1.0 - x[0])*(1.0 - x[0]))

    return 0

#*------------------------------------------------------------------* 
#*     FUNCTION callbackEvalG                                       *
#*------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "objGrad" is set in the KN_eval_result structure.
def callbackEvalG (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALGA:
        print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate gradient of nonlinear objective
    dTmp = x[1] - x[0]*x[0]
    evalResult.objGrad[0] = (-400.0 * dTmp * x[0]) - (2.0 * (1.0 - x[0]))
    evalResult.objGrad[1] = 200.0 * dTmp
        
    return 0

#*------------------------------------------------------------------* 
#*     FUNCTION callbackEvalH                                       *
#*------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "hess" and "hessVec" are set in the KN_eval_result structure.
def callbackEvalH (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
        print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x
    # Scale objective component of hessian by sigma
    sigma = evalRequest.sigma

    # Evaluate the hessian of the nonlinear objective.
    # Note: Since the Hessian is symmetric, we only provide the
    #       nonzero elements in the upper triangle (plus diagonal).
    #       These are provided in row major ordering as specified
    #       by the setting KN_DENSE_ROWMAJOR in "KN_set_cb_hess()".
    # Note: The Hessian terms for the quadratic constraints
    #       will be added internally by Knitro to form
    #       the full Hessian of the Lagrangian.
    evalResult.hess[0] = sigma * ( (-400.0 * x[1]) + (1200.0 * x[0]*x[0]) + 2.0) # (0,0)
    evalResult.hess[1] = sigma * (-400.0 * x[0]) # (0,1)
    evalResult.hess[2] = sigma * 200.0           # (1,1)
    
    return 0

#*------------------------------------------------------------------*
#*     main                                                         *
#*------------------------------------------------------------------*

# Create a new Knitro solver instance.
try:
    kc = KN_new ()
except:
    print ("Failed to find a valid license.")
    quit ()

# Illustrate how to override default options by reading from
# the knitro.opt file.
KN_load_param_file (kc, "knitro.opt")

# Initialize knitro with the problem definition.

# Add the 4 variables and set their bounds. 
# Note: any unset lower bounds are assumed to be
# unbounded below and any unset upper bounds are
# assumed to be unbounded above.
KN_add_vars (kc, 2)
KN_set_var_lobnds (kc, xLoBnds = [-KN_INFINITY, -KN_INFINITY]) # not necessary since infinite
KN_set_var_upbnds (kc, xUpBnds = [0.5, KN_INFINITY])

# Add the constraints and set their lower bounds
KN_add_cons (kc, 2)
KN_set_con_lobnds (kc, cLoBnds = [1.0, 0.0]) 

# Both constraints are quadratic so we can directly load all the
# structure for these constraints.

# First load quadratic structure x0*x1 for the first constraint
KN_add_con_quadratic_struct (kc, 0, 0, 1, 1.0)

# Load structure for the second constraint.  below we add the linear
# structure and the quadratic structure separately, though it
# is possible to add both together in one call to
# "KN_add_con_quadratic_struct()" since this api function also
# supports adding linear terms.

# Add linear term x0 in the second constraint
KN_add_con_linear_struct (kc, 1, 0, 1.0)
    
# Add quadratic term x1^2 in the second constraint
KN_add_con_quadratic_struct (kc, 1, 1, 1, 1.0)

# Add a callback function "callbackEvalF" to evaluate the nonlinear
# (non-quadratic) objective.  Note that the linear and
# quadratic terms in the objective could be loaded separately
# via "KN_add_obj_linear_struct()" / "KN_add_obj_quadratic_struct()".
# However, for simplicity, we evaluate the whole objective
# function through the callback.
cb = KN_add_eval_callback (kc, evalObj = True, funcCallback = callbackEvalF)

# Also add a callback function "callbackEvalG" to evaluate the
# objective gradient.  If not provided, Knitro will approximate
# the gradient using finite-differencing.  However, we recommend
# providing callbacks to evaluate the exact gradients whenever
# possible as this can drastically improve the performance of Knitro.
# We specify the objective gradient in "dense" form for simplicity.
# However for models with many constraints, it is important to specify
# the non-zero sparsity structure of the constraint gradients
# (i.e. Jacobian matrix) for efficiency (this is true even when using
# finite-difference gradients).
KN_set_cb_grad (kc, cb, objGradIndexVars = KN_DENSE, gradCallback = callbackEvalG)

# Add a callback function "callbackEvalH" to evaluate the Hessian
# (i.e. second derivative matrix) of the objective.  If not specified,
# Knitro will approximate the Hessian. However, providing a callback
# for the exact Hessian (as well as the non-zero sparsity structure)
# can greatly improve Knitro performance and is recommended if possible.
# Since the Hessian is symmetric, only the upper triangle is provided.
# Again for simplicity, we specify it in dense (row major) form.
KN_set_cb_hess (kc, cb, hessIndexVars1 = KN_DENSE_ROWMAJOR, hessCallback = callbackEvalH)

# specify that the user is able to provide evaluations
# of the hessian matrix without the objective component.
# turned off by default but should be enabled if possible.
KN_set_int_param (kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
    
# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

# Turn output off and use Interior/Direct algorithm
KN_set_int_param (kc, "outlev", KN_OUTLEV_NONE)
KN_set_int_param (kc, "algorithm", KN_ALG_BAR_DIRECT)

# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.

# First solve for the 6 different values of user option "bar_murule".
# This option handles how the barrier parameter is updated each
# iteration in the barrier/interior-point solver.
print ("Changing a user option and re-solving...")
for i in range (1,7):
    KN_set_int_param (kc, "bar_murule", i)
    # Reset original initial point
    KN_set_var_primal_init_values (kc, xInitVals = [-2.0, 1.0])
    nStatus = KN_solve (kc)
    if nStatus != 0:
        print ("  bar_murule=%d - Knitro failed to solve, status = %d" % (i, nStatus))
    else:
        print ("  bar_murule=%d - solved in %2d iters, %2d function evaluations, objective=%e" %
                (i, KN_get_number_iters (kc), KN_get_number_FC_evals (kc), KN_get_obj_value (kc)))

# Now solve for different values of the x0 upper bound.
# Continually relax the upper bound until it is no longer
# "active" (i.e. no longer restricting x0), at which point
# there is no more significant change in the optimal solution.
# Change to the active-set algorithm and do not reset the
# initial point, so the re-solves are "warm-started".
print ("\nChanging a variable bound and re-solving...")
KN_set_int_param (kc, "algorithm", KN_ALG_ACT_CG)
tmpbound = 0.5
i = 0
while True:
    if i > 0:
        # Modify bound for next solve.
        tmpbound += 0.1
        KN_set_var_upbnds (kc, 0, tmpbound)

    nStatus = KN_solve (kc)
    if nStatus != 0:
        print ("  x0 upper bound=%e - Knitro failed to solve, status = %d" % (tmpbound, nStatus))
    else:
        nStatus, objSol, x, lambda_ = KN_get_solution (kc)
        print ("  x0 upper bound=%e - solved in %2d iters, x0=%e, objective=%e" % (tmpbound, KN_get_number_iters (kc), x[0], objSol))

    i += 1
    if nStatus != 0 or x[0] < tmpbound - 1e-4:
        break
# Restore original value
KN_set_var_upbnds (kc, 0, 0.5)

# Now solve for different values of the c0 lower bound.
# Continually relax the lower bound until it is no longer
# "active" (i.e. no longer restricting c0), at which point
# there is no more significant change in the optimal solution.
print ("\nChanging a constraint bound and re-solving...")
tmpbound = 1.0
i = 0
while True:
    if i > 0:
        # Modify bound for next solve.
        tmpbound -= 0.1
        KN_set_con_lobnds (kc, 0, tmpbound)
    nStatus = KN_solve (kc)
    if nStatus != 0:
        print ("  c0 lower bound=%e - Knitro failed to solve, status = %d" % (tmpbound, nStatus))
    else:
        c0 = KN_get_con_values (kc, 0)
        print ("  c0 lower bound=%e - solved in %2d iters, c0=%e, objective=%e" %
                (tmpbound, KN_get_number_iters (kc), c0, KN_get_obj_value (kc)))
    i += 1
    if nStatus != 0 or c0 > tmpbound + 1e-4:
        break

# Delete the Knitro solver instance.
KN_free (kc)
