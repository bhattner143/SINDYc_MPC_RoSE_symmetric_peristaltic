#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Knitro to solve the following
# nonlinear optimization problem in the simplest way possible,
# without providing any derivative information.  If at all possible,
# derivative information should be provided as it will greatly
# improve the performance of Knitro.  See exampleNLP1.c to see the
# same model solved with the derivatives provided.
#
# This model is test problem HS15 from the Hock & Schittkowski
# collection.
#
# min   100 (x1 - x0^2)^2 + (1 - x0)^2
# s.t.  x0 x1 >= 1
#       x0 + x1^2 >= 0
#       x0 <= 0.5
#
# The standard start point (-2, 1) usually converges to the standard
# minimum at (0.5, 2.0), with final objective = 306.5.
# Sometimes the solver converges to another local minimum
# at (-0.79212, -1.26243), with final objective = 360.4.
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

# Initialize Knitro with the problem definition.

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
KN_add_cons(kc, m)
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
indexVar1 = 0
coef = 1.0
KN_add_con_linear_struct (kc, 1, 0, 1.0)

# Add quadratic term x1^2 in the second constraint
indexVar1 = 1
indexVar2 = 1
coef = 1.0
KN_add_con_quadratic_struct (kc, 1, 1, 1, 1.0)


# Add a callback function "callbackEvalF" to evaluate the nonlinear
# (non-quadratic) objective.  Note that the linear and
# quadratic terms in the objective could be loaded separately
# via "KN_add_obj_linear_struct()" / "KN_add_obj_quadratic_struct()".
# However, for simplicity, we evaluate the whole objective
# function through the callback.
cb = KN_add_eval_callback (kc, evalObj=True, funcCallback=callbackEvalF)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Set the non-default SQP algorithm, which typically converges in the
# fewest number of function evaluations.  This algorithm (or the
# active-set algorithm ("KN_ALG_ACT_CG") may be preferable for
# derivative-free optimization models with expensive function
# evaluations.
KN_set_int_param (kc, KN_PARAM_ALGORITHM, KN_ALG_ACT_SQP)

# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.
nStatus = KN_solve (kc)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ = KN_get_solution (kc)
print ("Optimal objective value  = %e" % objSol)
print ("Optimal x (with corresponding multiplier)")
for i in range (n):
    print ("  x[%d] = %e (lambda = %e)" % (i, x[i], lambda_[m+i]))
print ("Optimal constraint values (with corresponding multiplier)")
c = KN_get_con_values (kc)
for j in range (m):
    print ("  c[%d] = %e (lambda = %e)" % (i, c[i], lambda_[i]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error (kc))

# Delete the Knitro solver instance.
KN_free (kc)
