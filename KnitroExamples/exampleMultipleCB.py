#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Knitro to solve the following
# simple nonlinear optimization problem.  This model is test problem
# HS40 from the Hock & Schittkowski collection.
#
# max   x0*x1*x2*x3         (obj)
# s.t.  x0^3 + x1^2 = 1     (c0)
#       x0^2*x3 - x2 = 0    (c1)
#       x3^2 - x1 = 0       (c2)
#
# This is the same as exampleNLP2.c, but it demonstrates using
# multiple callbacks for the nonlinear evaluations and computing
# some gradients using finite-differencs, while others are provided
# in callback routines.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

# A pointer to this class will be passed thru Knitro to
# reach the evaluation callback routines.
class UserEvalType:
    def __init__ (self, xIndices):
        self.xIndices = xIndices

#*------------------------------------------------------------------* 
#*     FUNCTION EVALUATION CALLBACKS                                *
#*------------------------------------------------------------------*

# The signature of this function matches KN_eval_callback in knitro.py.
# Only "obj" is set in the KN_eval_result structure.
def callbackEvalObj (kc, cb, evalRequest, evalResult, userParams):
    xind = userParams.xIndices
 
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalObj incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x
    
    # Evaluate nonlinear term in objective
    evalResult.obj = x[xind[0]]*x[xind[1]]*x[xind[2]]*x[xind[3]]

    return 0

# The signature of this function matches KN_eval_callback in knitro.py.
# Only "c0" is set in the KN_eval_result structure.
def callbackEvalC0 (kc, cb, evalRequest, evalResult, userParams):
    xind = userParams.xIndices
    
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalC0 incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear terms in constraint, c0    
    evalResult.c[0] = x[xind[0]]*x[xind[0]]*x[xind[0]]
        
    return 0

# The signature of this function matches KN_eval_callback in knitro.py.
# Only "c1" is set in the KN_eval_result structure.
def callbackEvalC1 (kc, cb, evalRequest, evalResult, userParams):
    xind = userParams.xIndices
    
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalC1 incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear terms in constraint, c1
    evalResult.c[0] = x[xind[0]]*x[xind[0]]*x[xind[3]]
        
    return 0

#*------------------------------------------------------------------* 
#*     GRADIENT EVALUATION CALLBACKS                                *
#*------------------------------------------------------------------*

# The signature of this function matches KN_eval_callback in knitro.py.
# Only "objGrad" is set in the KN_eval_result structure.
def callbackEvalObjGrad (kc, cb, evalRequest, evalResult, userParams):
    xind = userParams.xIndices
    
    if evalRequest.type != KN_RC_EVALGA:
        print ("*** callbackEvalObjGrad incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x
 
    # Evaluate nonlinear terms in objective gradient
    evalResult.objGrad[xind[0]] = x[xind[1]]*x[xind[2]]*x[xind[3]]
    evalResult.objGrad[xind[1]] = x[xind[0]]*x[xind[2]]*x[xind[3]]
    evalResult.objGrad[xind[2]] = x[xind[0]]*x[xind[1]]*x[xind[3]]
    evalResult.objGrad[xind[3]] = x[xind[0]]*x[xind[1]]*x[xind[2]]
    
    return 0

# The signature of this function matches KN_eval_callback in knitro.py.
# Only gradient of c0 is set in the KN_eval_result structure.
def callbackEvalC0Grad (kc, cb, evalRequest, evalResult, userParams):
    xind = userParams.xIndices
    
    if evalRequest.type != KN_RC_EVALGA:
        print ("*** callbackEvalC0Grad incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear terms in c0 constraint gradients
    evalResult.jac[0] = 3.0*x[xind[0]]*x[xind[0]] #* derivative of x0^3 term  wrt x0
        
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
xIndices = KN_add_vars (kc, 4)
for x in xIndices:
    KN_set_var_primal_init_values (kc, x, 0.8)
    
# Add the constraints and set the rhs and coefficients
cIndices = KN_add_cons (kc, 3)
KN_set_con_eqbnds (kc, cIndices[0], 1.0)
KN_set_con_eqbnds (kc, cIndices[1], 0.0)
KN_set_con_eqbnds (kc, cIndices[2], 0.0)

# Coefficients for 2 linear terms
lconIndexCons = [1, 2]
lconIndexVars = [2, 1]
lconCoefs = [-1.0, -1.0]
KN_add_con_linear_struct (kc, lconIndexCons, lconIndexVars, lconCoefs)

# Coefficients for 2 quadratic terms

#* x1^2 term in c0
qconIndexCons = [0]
qconIndexVars1 = [1]
qconIndexVars2 = [1]
qconCoefs = [1.0]

#* x3^2 term in c2
qconIndexCons += [2]
qconIndexVars1 += [3]
qconIndexVars2 += [3]
qconCoefs += [1.0]

KN_add_con_quadratic_struct (kc, qconIndexCons, qconIndexVars1, qconIndexVars2, qconCoefs)

# Add separate callbacks.

# Set callback data for nonlinear objective term.    
cbObj = KN_add_eval_callback (kc, evalObj = True, funcCallback = callbackEvalObj)
KN_set_cb_grad (kc, cbObj, objGradIndexVars = xIndices, gradCallback = callbackEvalObjGrad)
    
# Set callback data for nonlinear constraint 0 term.
cbC0 = KN_add_eval_callback (kc, indexCons = cIndices[0], funcCallback = callbackEvalC0)
indexCons = cIndices[0]  # constraint c0
indexVars = xIndices[0]  # variable x0
KN_set_cb_grad (kc, cbC0, jacIndexCons = indexCons, jacIndexVars = indexVars, gradCallback = callbackEvalC0Grad)

# Set callback data for nonlinear constraint 1 term
cbC1 = KN_add_eval_callback (kc, indexCons = cIndices[1], funcCallback = callbackEvalC1)
indexCons = [cIndices[1], cIndices[1]]  # constraint c1
indexVars = [xIndices[0], xIndices[0]]  # variables x0 and x3
KN_set_cb_grad(kc, cbC1, jacIndexCons = indexCons, jacIndexVars = indexVars)
# This one will be approximated via forward finite differences.
KN_set_cb_gradopt (kc, cbC1, KN_GRADOPT_FORWARD)

# Demonstrate passing a userParams structure to the evaluation
# callbacks.  Here we pass back the variable indices set from
# KN_add_vars() for use in the callbacks. Here we pass the same
# userParams structure to each callback but we could define different
# userParams for different callbacks.  This could be useful, for
# instance, if different callbacks operate on different sets of
# variables.
userEval = UserEvalType (xIndices)
KN_set_cb_user_params (kc, cbObj, userEval)
KN_set_cb_user_params (kc, cbC0, userEval)
KN_set_cb_user_params (kc, cbC1, userEval)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MAXIMIZE)

# Approximate hessian using BFGS
KN_set_int_param (kc, "hessopt", KN_HESSOPT_BFGS)

# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.
nStatus = KN_solve (kc)

print ()
print ("Knitro converged with final status = %d" % nStatus)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ =  KN_get_solution (kc)
print ("  optimal objective value  = %e" % objSol)
print ("  optimal primal values x  = (%e, %e, %e)" % (x[0], x[1], x[2]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error (kc))

# Delete the Knitro solver instance.
KN_free (kc)
