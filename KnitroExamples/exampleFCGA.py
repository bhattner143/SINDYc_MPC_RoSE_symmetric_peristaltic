#*******************************************************/
#* Copyright (c) 2019 by Artelys                       */
#* All Rights Reserved                                 */
#*******************************************************/

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Knitro example defining a single callback function for function
#  and gradient (i.e. EVALFCGA) requests, and a second callback for
#  Hessian (i.e. EVALH / EVALHV) requests.  This model is test
#  problem HS40 from the Hock & Schittkowski collection. 
#
#  max   x0*x1*x2*x3         (obj)
#  s.t.  x0^3 + x1^2 = 1     (c0)
#        x0^2*x3 - x2 = 0    (c1)
#        x3^2 - x1 = 0       (c2)
#
#  See exampleNLP2.c to see how this is solved in the standard
#  way with separate callbacks for functions and gradients.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

#*------------------------------------------------------------------*/ 
#*     FUNCTION callbackEvalFCGA                                    */
#*------------------------------------------------------------------*/
# The signature of this function matches KN_eval_callback in knitro.h.
# To compute the functions and gradient together, set "obj", "c",
# "objGrad" and "jac" in the KN_eval_result structure when there
# is a function+gradient evaluation request (i.e. EVALFCGA).
#
# NOTE: It is generally more efficient and recommended to have
# separate callback routines for functions and gradients since a
# gradient evaluation is not always needed for every function
# evaluation.  However, in some cases, it may be more convenient
# to compute them together if most of the work for computing the
# gradients is already done for the function evaluation.
def callbackEvalFCGA (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALFCGA:
        print ("*** callbackEvalFCGA incorrectly called with eval type %d" % evalRequest.type)
        return -1

    x = evalRequest.x
    
    # Evaluate nonlinear term in objective
    evalResult.obj = x[0]*x[1]*x[2]*x[3]
    
    # Evaluate nonlinear terms in constraints
    evalResult.c[0] = x[0]*x[0]*x[0]
    evalResult.c[1] = x[0]*x[0]*x[3]
    
    # Evaluate nonlinear term in objective gradient
    evalResult.objGrad[0] = x[1]*x[2]*x[3]
    evalResult.objGrad[1] = x[0]*x[2]*x[3]
    evalResult.objGrad[2] = x[0]*x[1]*x[3]
    evalResult.objGrad[3] = x[0]*x[1]*x[2]
        
    # Evaluate nonlinear terms in constraint gradients (Jacobian)
    evalResult.jac[0] = 3.0*x[0]*x[0] # derivative of x0^3 term  wrt x0
    evalResult.jac[1] = 2.0*x[0]*x[3] # derivative of x0^2*x3 term  wrt x0
    evalResult.jac[2] = x[0]*x[0]     # derivative of x0^2*x3 terms wrt x3
    
    return 0


#*------------------------------------------------------------------*/ 
#*     FUNCTION callbackEvalH                                       */
#*------------------------------------------------------------------*/
# The signature of this function matches KN_eval_callback in knitro.h.
# Only "hess" and "hessVec" are set in the KN_eval_result structure.
def callbackEvalH (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
        print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
        return -1

    x = evalRequest.x
    lambda_ = evalRequest.lambda_
    # Scale objective component of Hessian by sigma
    sigma = evalRequest.sigma

    # Evaluate nonlinear terms in the Hessian of the Lagrangian.
    # Note: If sigma=0, some computations can be avoided.
    if sigma > 0.0:  # Evaluate the full Hessian of the Lagrangian
        evalResult.hess[0] = lambda_[0]*6.0*x[0] + lambda_[1]*2.0*x[3]
        evalResult.hess[1] = sigma*x[2]*x[3]
        evalResult.hess[2] = sigma*x[1]*x[3]
        evalResult.hess[3] = sigma*x[1]*x[2] + lambda_[1]*2.0*x[0]
        evalResult.hess[4] = sigma*x[0]*x[3]
        evalResult.hess[5] = sigma*x[0]*x[2]
        evalResult.hess[6] = sigma*x[0]*x[1]
    else:            # sigma=0, do not include objective component
        evalResult.hess[0] = lambda_[0]*6.0*x[0] + lambda_[1]*2.0*x[3]
        evalResult.hess[1] = 0.0
        evalResult.hess[2] = 0.0
        evalResult.hess[3] = lambda_[1]*2.0*x[0]
        evalResult.hess[4] = 0.0
        evalResult.hess[5] = 0.0
        evalResult.hess[6] = 0.0
    
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

# Initialize Knitro with the problem definition.

# Add the variables and specify initial values for them. 
# Note: any unset lower bounds are assumed to be
# unbounded below and any unset upper bounds are
# assumed to be unbounded above.
vars = KN_add_vars (kc, 4)
for x in vars:
    KN_set_var_primal_init_values (kc, x, 0.8)
    
# Add the constraints and set the rhs and coefficients
KN_add_cons (kc, 3)
KN_set_con_eqbnds (kc, cEqBnds = [1.0, 0.0, 0.0]);

# Coefficients for 2 linear terms
lconIndexCons = [1, 2]
lconIndexVars = [2, 1]
lconCoefs = [-1.0, -1.0]
KN_add_con_linear_struct (kc, lconIndexCons, lconIndexVars, lconCoefs)

# Coefficients for 2 quadratic terms

# x1^2 term in c0
qconIndexCons = [0]
qconIndexVars1 = [1]
qconIndexVars2 = [1]
qconCoefs = [1.0]

# x3^2 term in c2
qconIndexCons += [2]
qconIndexVars1 += [3]
qconIndexVars2 += [3]
qconCoefs += [1.0]

KN_add_con_quadratic_struct (kc, qconIndexCons, qconIndexVars1, qconIndexVars2, qconCoefs)

# Add callback to evaluate nonlinear (non-quadratic) terms in the model:
#    x0*x1*x2*x3  in the objective 
#    x0^3         in first constraint c0
#    x0^2*x3      in second constraint c1
cIndices = [0, 1]
cb = KN_add_eval_callback (kc, evalObj = True, indexCons = cIndices, funcCallback = callbackEvalFCGA)

# Set obj. gradient and nonlinear jac provided through callbacks.
# Mark objective gradient as dense, and provide non-zero sparsity
# structure for constraint Jacobian terms. Set the gradient
# evaluations to use the same callback routine as used for
# function evaluations.
cbjacIndexCons = [0, 1, 1]
cbjacIndexVars = [0, 0, 3]
error = KN_set_cb_grad (kc, cb, objGradIndexVars = KN_DENSE, jacIndexCons = cbjacIndexCons, jacIndexVars = cbjacIndexVars)
    
# Set nonlinear Hessian provided through callbacks. Since the
# Hessian is symmetric, only the upper triangle is provided.
# The upper triangular Hessian for nonlinear callback structure is:
#
#  lambda0*6*x0 + lambda1*2*x3     x2*x3    x1*x3    x1*x2 + lambda1*2*x0
#               0                    0      x0*x3         x0*x2
#                                             0           x0*x1
#                                                          0
#  (7 nonzero elements)
cbhessIndexVars1 = [0, 0, 0, 0, 1, 1, 2]
cbhessIndexVars2 = [0, 1, 2, 3, 2, 3, 3]
KN_set_cb_hess (kc, cb, hessIndexVars1 = cbhessIndexVars1, hessIndexVars2 = cbhessIndexVars2, hessCallback = callbackEvalH)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MAXIMIZE)

# Set option to print output after every iteration.
KN_set_int_param (kc, KN_PARAM_OUTLEV, KN_OUTLEV_ITER)

# Set option to tell Knitro that the gradients are being provided
# with the functions in one callback.
KN_set_int_param (kc, KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES)

# Solve the problem.
#
# Return status codes are defined in "knitro.h" and described
# in the Knitro manual.
nStatus = KN_solve (kc)

print ()
print ("Knitro converged with final status = %d" % nStatus)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ =  KN_get_solution (kc)
print ("  optimal objective value  = %e" % objSol)
print ("  optimal primal values x  = (%e, %e, %e, %e)" % (x[0], x[1], x[2], x[3]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error (kc))

# Delete the Knitro solver instance.
KN_free (kc)
