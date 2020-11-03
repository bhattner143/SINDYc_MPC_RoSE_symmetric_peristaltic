#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  This example demonstrates how to use Knitro to solve the following
#  simple quadratically constrained quadratic programming problem (QCQP).  
#
#  min   1000 - x0^2 - 2 x1^2 - x2^2 - x0 x1 - x0 x2
#  s.t.  8 x0 + 14 x1 + 7 x2 = 56
#        x0^2 + x1^2 + x2^2 >= 25
#        x0 >= 0, x1 >= 0, x2 >= 0
#
#  The start point (2, 2, 2) converges to the minimum at (0, 0, 8),
#  with final objective = 936.0.  From a different start point,
#  Knitro may converge to an alternate local solution at (7, 0, 0),
#  with objective = 951.0.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

# Create a new Knitro solver instance.
try:
    kc = KN_new ()
except:
    print ("Failed to find a valid license.")

# Illustrate how to override default options by reading from
# the knitro.opt file.
KN_load_param_file (kc, "knitro.opt")

# Initialize Knitro with the problem definition.

# Add the variables and set their bounds and initial values.
# Note: unset bounds assumed to be infinite.
KN_add_vars (kc, 3)
KN_set_var_lobnds (kc, xLoBnds = [0, 0, 0])
KN_set_var_primal_init_values (kc, xInitVals = [2.0, 2.0, 2.0])

# Add the constraints and set their bounds.
KN_add_cons (kc, 2)
KN_set_con_eqbnds (kc, 0, 56.0)
KN_set_con_lobnds (kc, 1, 25.0)

# Add coefficients for linear constraint.
lconIndexVars = [  0,    1,   2]
lconCoefs     = [8.0, 14.0, 7.0]
KN_add_con_linear_struct (kc, 0, lconIndexVars, lconCoefs)

# Add coefficients for quadratic constraint
qconIndexVars1 = [  0,   1,   2]
qconIndexVars2 = [  0,   1,   2]
qconCoefs      = [1.0, 1.0, 1.0]
KN_add_con_quadratic_struct (kc, 1, qconIndexVars1, qconIndexVars2, qconCoefs)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Add constant value to the objective.
KN_add_obj_constant (kc, 1000.0) 

# Set quadratic objective structure.
qobjIndexVars1 = [   0,    1,    2,    0,    0]
qobjIndexVars2 = [   0,    1,    2,    1,    2]
qobjCoefs      = [-1.0, -2.0, -1.0, -1.0, -1.0]

KN_add_obj_quadratic_struct (kc, qobjIndexVars1, qobjIndexVars2, qobjCoefs)

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
