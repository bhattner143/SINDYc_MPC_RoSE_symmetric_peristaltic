#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  This example demonstrates how to use Knitro to solve the following
#  simple quadratic programming problem (QP).  
#
#     minimize     0.5*(x0^2+x1^2+x2^2) + 11*x0 + x2
#     subject to   -6*x2  <= 5
#                 0 <= x0 
#                 0 <= x1
#                -3 <= x2 <= 2
#
#  The optimal solution is:
#     obj=-0.4861   x=[0,0,-5/6]
#
#  The purpose is to illustrate how to invoke Knitro using the C
#  language API.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

# Used to specify whether linear and quadratic objective
# terms are loaded separately or together in this example.
bSeparate = True

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
# Note: unset bounds assumed to be infinite.
KN_add_vars (kc, 3)
KN_set_var_lobnds (kc, xLoBnds = [0.0, 0.0, -3.0])
KN_set_var_upbnds (kc, 2, 2.0)

# Add the constraint and set the bound and coefficient.
KN_add_cons (kc, 1)
KN_set_con_upbnds (kc, 0, 5.0)
KN_add_con_linear_struct (kc, 0, 2, -6.0)

# Set the coefficients for the objective -
# can either set linear and quadratic objective structure
# separately or all at once.  We show both cases below.
# Change the value of "bSeparate" to try both cases.

if (bSeparate):
    # Set linear and quadratic objective structure separately.
    # First set linear objective structure.
    lobjIndexVars = [0, 2]
    lobjCoefs = [11.0, 1.0]
    KN_add_obj_linear_struct (kc, lobjIndexVars, lobjCoefs)
    # Now set quadratic objective structure.
    qobjIndexVars1 = [0, 1, 2]
    qobjIndexVars2 = [0, 1, 2]
    qobjCoefs = [0.5, 0.5, 0.5]
    KN_add_obj_quadratic_struct (kc, qobjIndexVars1, qobjIndexVars2, qobjCoefs)
else:
    # Example of how to set linear and quadratic objective
    # structure at once. Setting the 2nd variable index in a
    # quadratic term to be negative, treats it as a linear term.
    indexVars1 = [0, 1, 2, 0, 2]
    indexVars2 = [0, 1, 2, -1, -1]  # -1 for linear coefficients
    objCoefs = [0.5, 0.5, 0.5, 11.0, 1.0]
    KN_add_obj_quadratic_struct (kc, indexVars1, indexVars2, objCoefs)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Enable iteration output and crossover procedure to try to
# get more solution precision
KN_set_int_param (kc, KN_PARAM_OUTLEV, KN_OUTLEV_ITER)
KN_set_int_param (kc, KN_PARAM_BAR_MAXCROSSIT, 5)
    
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
