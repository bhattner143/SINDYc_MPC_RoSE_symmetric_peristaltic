#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  This example demonstrates how to use Knitro MPS reader
#  load an optimization problem from a MPS file and resolve it
#  with Knitro.
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *

# Path to MPS file.
mps_file = "lp.mps"

# Create new instance of Knitro.
kc = KN_new()

# Load MPS file inside Knitro.
KN_load_mps_file(kc, mps_file)

# Solve the problem.
#
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.
nStatus = KN_solve(kc)

print ()
print ("Knitro converged with final status = %d" % nStatus)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ =  KN_get_solution(kc)
print ("  optimal objective value  = %e" % objSol)
print ("  optimal primal values x  = (%e, %e, %e, %e)" % (x[0], x[1], x[2], x[3]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error(kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error(kc))

# Delete the Knitro solver instance.
KN_free(kc)
