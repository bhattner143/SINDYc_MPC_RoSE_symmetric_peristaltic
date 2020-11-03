#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains routines to implement problemDef.h for
# a simple nonlinear least-squares problem.
#
# min   ( x0*1.309^x1 - 2.138 )^2 + ( x0*1.471^x1 - 3.421 )^2 + ( x0*1.49^x1 - 3.597 )^2
#        + ( x0*1.565^x1 - 4.34 )^2 + ( x0*1.611^x1 - 4.882 )^2 + ( x0*1.68^x1-5.66 )^2  
#
# The standard start point (1.0, 5.0) usually converges to the standard
# minimum at (0.76886, 3.86041), with final objective = 0.00216.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *
import math

#*------------------------------------------------------------------* 
#*     FUNCTION pow                                                 *
#*------------------------------------------------------------------*
# This function mimics the behaviour of the std:pow function in C

#*------------------------------------------------------------------* 
#*     FUNCTION callbackEvalR                                       *
#*------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "rsd" is set in the KN_eval_result structure.
def callbackEvalR (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALR:
        print ("*** callbackEvalR incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Clamp x1 to 1000 so as to avoid overflow
    if x[1] > 1000.0:
        x[1] = 1000.0

    # Evaluate nonlinear residual components
    evalResult.rsd[0] = x[0] * math.pow(1.309, x[1])
    evalResult.rsd[1] = x[0] * math.pow(1.471, x[1]) 
    evalResult.rsd[2] = x[0] * math.pow(1.49, x[1])
    evalResult.rsd[3] = x[0] * math.pow(1.565, x[1])
    evalResult.rsd[4] = x[0] * math.pow(1.611, x[1])
    evalResult.rsd[5] = x[0] * math.pow(1.68, x[1])

    return 0

#*------------------------------------------------------------------* 
#*     FUNCTION callbackEvalRJ                                      *
#*------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "rsdJac" is set in the KN_eval_result structure.
def callbackEvalRJ (kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALRJ:
        print ("*** callbackEvalRJ incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Clamp x1 to 1000 so as to avoid overflow
    if x[1] > 1000.0:
        x[1] = 1000.0

    # Evaluate non-zero residual Jacobian elements (row major order).
    evalResult.rsdJac[0] = math.pow(1.309, x[1])
    evalResult.rsdJac[1] = x[0] * math.log(1.309) * math.pow(1.309, x[1])
    evalResult.rsdJac[2] = math.pow(1.471, x[1])
    evalResult.rsdJac[3] = x[0] * math.log(1.471) * math.pow(1.471, x[1])
    evalResult.rsdJac[4] = math.pow(1.49, x[1])
    evalResult.rsdJac[5] = x[0] * math.log(1.49) * math.pow(1.49, x[1])
    evalResult.rsdJac[6] = math.pow(1.565, x[1])
    evalResult.rsdJac[7] = x[0] * math.log(1.565) * math.pow(1.565, x[1])
    evalResult.rsdJac[8] = math.pow(1.611, x[1])
    evalResult.rsdJac[9] = x[0] * math.log(1.611) * math.pow(1.611, x[1])
    evalResult.rsdJac[10] = math.pow(1.68, x[1])
    evalResult.rsdJac[11] = x[0] * math.log(1.68) * math.pow(1.68, x[1])
        
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

# Add the variables/parameters.
# Note: Any unset lower bounds are assumed to be
# unbounded below and any unset upper bounds are
# assumed to be unbounded above.
n = 2 # # of variables/parameters    
KN_add_vars (kc, n)
    
# In order to prevent the possiblity of numerical
# overflow from very large numbers, we set a
# reasonable upper bound on variable x[1] and set the
# "honorbnds" option for this variable to enforce
# that all trial x[1] values satisfy this bound.
KN_set_var_upbnds (kc, 1, 100.0)
KN_set_var_honorbnds (kc, 1, KN_HONORBNDS_ALWAYS)

# Add the residuals.
m = 6 # # of residuals
KN_add_rsds (kc, m)
    
# Set the array of constants in the residuals
KN_add_rsd_constants (kc, constants = [-2.138, -3.421, -3.597, -4.34, -4.882, -5.66])

# Add a callback function "callbackEvalR" to evaluate the nonlinear
# residual components.  Note that the constant terms are added
# separately above, and will not be included in the callback.
cb = KN_add_lsq_eval_callback (kc, rsdCallback = callbackEvalR)
    
# Also add a callback function "callbackEvalRJ" to evaluate the
# Jacobian of the residuals.  If not provided, Knitro will approximate
# the residual Jacobian using finite-differencing.  However, we recommend
# providing callbacks to evaluate the exact Jacobian whenever
# possible as this can drastically improve the performance of Knitro.
# We specify the residual Jacobian in "dense" row major form for simplicity.
# However for models with many sparse residuals, it is important to specify
# the non-zero sparsity structure of the residual Jacobian for efficiency
# (this is true even when using finite-difference gradients).
KN_set_cb_rsd_jac (kc, cb, jacIndexRsds = KN_DENSE_ROWMAJOR, rsdJacCallback = callbackEvalRJ)
    
# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.

nRC = KN_solve (kc)

if nRC != 0:
    print ("Knitro failed to solve the problem, status = %d" % nRC)
else:
    # An example of obtaining solution information. 
    # Return status codes are defined in "knitro.py" and described
    # in the Knitro manual.
    nStatus, obj, x, lambda_ = KN_get_solution (kc)    
    print ("Knitro successful. The optimal solution is:")
    for i in range (n):
        print ("x[%d]=%e" % (i, x[i]))

# Delete the knitro solver instance.
KN_free (kc)
