#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Knitro to solve the following
# simple mathematical program with equilibrium/complementarity
# constraints (MPEC/MPCC).  
#
#  min   (x0 - 5)^2 + (2 x1 + 1)^2
#  s.t.  -1.5 x0 + 2 x1 + x2 - 0.5 x3 + x4 = 2
#        x2 complements (3 x0 - x1 - 3)
#        x3 complements (-x0 + 0.5 x1 + 4)
#        x4 complements (-x0 - x1 + 7)
#        x0, x1, x2, x3, x4 >= 0
#      
# The complementarity constraints must be converted so that one
# nonnegative variable complements another nonnegative variable.
#
#  min   (x0 - 5)^2 + (2 x1 + 1)^2
#  s.t.  -1.5 x0 + 2 x1 + x2 - 0.5 x3 + x4 = 2   (c0)
#        3 x0 - x1 - 3 - x5 = 0                  (c1)
#        -x0 + 0.5 x1 + 4 - x6 = 0               (c2)
#        -x0 - x1 + 7 - x7 = 0                   (c3)
#        x2 complements x5
#        x3 complements x6
#        x4 complements x7
#        x0, x1, x2, x3, x4, x5, x6, x7 >= 0
#
# The solution is (1, 0, 3.5, 0, 0, 0, 3, 6), with objective value 17.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

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

# Add the variables and set their bounds and initial values.
# Note: unset bounds assumed to be infinite.
KN_add_vars (kc, 8)
KN_set_var_lobnds (kc, xLoBnds = [0]*8)
KN_set_var_primal_init_values (kc, xInitVals = [0]*8)
    
# Add the constraints and set their bounds.
KN_add_cons (kc, 4)
KN_set_con_eqbnds (kc, cEqBnds = [2, 3, -4, -7])

# Add coefficients for all linear constraints at once.

# c0
lconIndexCons = [0, 0, 0, 0, 0]
lconIndexVars = [0, 1, 2, 3, 4]
lconCoefs = [-1.5, 2.0, 1.0, -0.5, 1.0]

# c1
lconIndexCons += [1, 1, 1]
lconIndexVars += [0, 1, 5]
lconCoefs += [3.0, -1.0, -1.0]

# c2
lconIndexCons += [2, 2, 2]
lconIndexVars += [0, 1, 6]
lconCoefs += [-1.0, 0.5, -1.0]

# c3
lconIndexCons += [3, 3, 3]
lconIndexVars += [0, 1, 7]
lconCoefs += [-1.0, -1.0, -1.0]

KN_add_con_linear_struct (kc, lconIndexCons, lconIndexVars, lconCoefs)

# Note that the objective (x0 - 5)^2 + (2 x1 + 1)^2 when
# expanded becomes:
#    x0^2 + 4 x1^2 - 10 x0 + 4 x1 + 26 

# Add quadratic coefficients for the objective
qobjIndexVars1 = [0, 1]
qobjIndexVars2 = [0, 1]
qobjCoefs = [1.0, 4.0]
KN_add_obj_quadratic_struct (kc, qobjIndexVars1, qobjIndexVars2, qobjCoefs)

# Add linear coefficients for the objective
lobjIndexVars = [0, 1]
lobjCoefs = [-10.0, 4.0]
KN_add_obj_linear_struct (kc, lobjIndexVars, lobjCoefs)

# Add constant to the objective
KN_add_obj_constant (kc, 26.0)
    
# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Now add the complementarity constraints
ccTypes = [KN_CCTYPE_VARVAR, KN_CCTYPE_VARVAR, KN_CCTYPE_VARVAR]
indexComps1 = [2, 3, 4]
indexComps2 = [5, 6, 7]
KN_set_compcons (kc, ccTypes, indexComps1, indexComps2)
    
# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.
nStatus = KN_solve (kc)
print ("Knitro converged with final status = %d" % nStatus)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ = KN_get_solution (kc)
print ("  optimal objective value  = %e" % objSol)
print ("  optimal primal values x0=%e" % x[0])
print ("                        x1=%e" % x[1])
print ("                        x2=%e complements x5=%e" % (x[2], x[5]))
print ("                        x3=%e complements x6=%e" % (x[3], x[6]))
print ("                        x4=%e complements x7=%e" % (x[4], x[7]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error (kc))    

# Delete the Knitro solver instance.
KN_free (kc)
