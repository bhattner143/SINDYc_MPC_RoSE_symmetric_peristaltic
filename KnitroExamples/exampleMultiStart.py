#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to apply the Knitro multi-start
# procedure to solve the following simple nonlinear optimization
# problem.  This model is test problem HS15 from the Hock &
# Schittkowski collection. 
#
# min   100 (x1 - x0^2)^2 + (1 - x0)^2
# s.t.  x0 x1 >= 1
#       x0 + x1^2 >= 0
#       x0 <= 0.5
#
# Some solves should converge to the minimum at (0.5, 2.0),
# with final objective = 306.5, while others may converge to
# another local minimum at (-0.79212, -1.26243), with final
# objective = 360.4.
#
# The example also shows how to set a callback function to
# perform some user-defined task after each multi-start solve.
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
#*     FUNCTION callbackMSProcess                                   *
#*------------------------------------------------------------------*
# The signature of this function matches KN_user_callback in knitro.py.
# Argument "kcSub" is the context pointer for the last multi-start
# subproblem solved inside Knitro. 
def  callbackMSProcess (kcSub, x, lambda_, userParams):
    # Print solution of the just completed multi-start solve.
    
    n = KN_get_number_vars (kcSub)
    print ("callbackMSProcess: ")
    print ("    Last solution: obj=%e" % KN_get_obj_value (kcSub))
    for i in range (n):
        print ("                   x[%d]=%e" % (i, x[i]))
    
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

# Initialize knitro with the problem definition.

# Add the variables and set their bounds. 
# Note: any unset lower bounds are assumed to be
# unbounded below and any unset upper bounds are
# assumed to be unbounded above.
n = 2
KN_add_vars (kc, n)
KN_set_var_lobnds (kc, xLoBnds = [-KN_INFINITY, -KN_INFINITY]) # not necessary since infinite
KN_set_var_upbnds (kc, xUpBnds = [0.5, KN_INFINITY])
# Define an initial point.  If not set, Knitro will generate one.
KN_set_var_primal_init_values (kc, xInitVals = [-2.0, 1.0])
    
# Add the constraints and set their lower bounds
m = 2
KN_add_cons (kc, m)
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
        
# Set minimize or maximize (if not set, assumed minimze)
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Example of how to register a callback function that performs some
# task after each multistart solve.
KN_set_ms_process_callback (kc, callbackMSProcess)
#    
# Disable automatic scaling.
KN_set_int_param (kc, KN_PARAM_SCALE, KN_SCALE_NO)
    
# Enable multi-start
KN_set_int_param (kc, KN_PARAM_MULTISTART, KN_MULTISTART_YES)

# Perform multistart in parallel using max number of available threads
try:
    import multiprocessing
    nThreads = multiprocessing.cpu_count()
    if nThreads > 1:
        print ("Running Knitro multistart in parallel with %d threads." % nThreads)
        KN_set_int_param (kc, KN_PARAM_PAR_MSNUMTHREADS, nThreads)
except:
    pass

# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.

nStatus = KN_solve (kc)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ =  KN_get_solution (kc)
print ("Optimal objective value  = %e" % objSol)
print ("Optimal x (with corresponding multiplier)")
for i in range (n):
    print ("  x[%d] = %e (lambda = %e)" % (i, x[i], lambda_[m+i]))
print ("Optimal constraint values (with corresponding multiplier)")
c = KN_get_con_values (kc)
for i in range (m):
    print ("  c[%d] = %e (lambda = %e)" % (i, c[i], lambda_[i]))
print ("Feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("Optimality violation     = %e" % KN_get_abs_opt_error (kc))

# Delete the Knitro solver instance.
KN_free (kc)
