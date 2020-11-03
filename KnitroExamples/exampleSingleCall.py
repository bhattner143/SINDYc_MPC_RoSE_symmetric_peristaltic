#*******************************************************
#* Copyright (c) 2019 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  This example demonstrates how to use Artelys Knitro single call
#  function (optimize) to quickly solve different types of
#  optimization problems.
#
#  Those problems can also be solved using the Python API directly as
#  presented in the examples LP1, MPEC1, NLP1, NLP1NoDerivs, FCGA
#  and MINLP1.
#
#  A detailed description of the data structure available to pass
#  information to the single call function is presented in knitro.py
#  or directly in Artelys Knitro documentation.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from knitro import *
import math

### Single call function LP1 example
def example_single_call_LP1():
    # ExampleLP1
    # min    -4*x0 - 2*x1
    # s.t.   x0 + x1 + x2        = 5
    #        2*x0 + 0.5*x1 + x3  = 8
    #        0 <= (x0, x1, x2, x3)

    # Define the variables information
    variables = Variables(nV=4, xLoBnds=[0,0,0,0])

    # Define the objective information
    # Default objGoal is set to 'minimize'
    objective = Objective(objLinear=[[0, 1], [-4, -2]])

    # Define the constraints information
    constraints = Constraints(nC=2,
                              cLinear=[[0, 0, 0, 1, 1, 1],
                                       [0, 1, 2, 0, 1, 3],
                                       [1., 1., 1., 2., 0.5, 1.]],
                              cEqBnds=[5., 8.])

    # Solve the problem
    solution = optimize(variables=variables,
                        objective=objective,
                        constraints=constraints)


### Single call function MPEC1 example
def example_single_call_MPEC1():
    # ExampleMPEC1
    #  min   (x0 - 5)^2 + (2 x1 + 1)^2
    #  s.t.  -1.5 x0 + 2 x1 + x2 - 0.5 x3 + x4 = 2
    #        x2 complements (3 x0 - x1 - 3)
    #        x3 complements (-x0 + 0.5 x1 + 4)
    #        x4 complements (-x0 - x1 + 7)
    #        x0, x1, x2, x3, x4 >= 0

    # Define the variables information
    variables = Variables(nV=8,
                          xLoBnds=[0]*8,
                          xInitVals=[0]*8)

    # Define the objective information
    objective = Objective(objConstant=26.0,
                          objLinear=[[0, 1], [-10.0, 4.0]],
                          objQuadratic=[[0, 1], [0, 1], [1.0, 4.0]])

    # Define the constraints information
    constraints = Constraints(nC=4,
                              cLinear=[[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                      [0, 1, 2, 3, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7],
                                      [-1.5, 2.0, 1.0, -0.5, 1.0, 3.0, -1.0, -1.0, -1.0, 0.5, -1.0, -1.0, -1.0, -1.0]],
                              cEqBnds=[2, 3, -4, -7],
                              cNames=["con0","con1","con2","con3"])

    # Define the complementarity constraints information
    compConstraints = ComplementarityConstraints(ccTypes=[KN_CCTYPE_VARVAR,
                                                          KN_CCTYPE_VARVAR,
                                                          KN_CCTYPE_VARVAR],
                                                 indexComps1=[2, 3, 4],
                                                 indexComps2=[5, 6, 7],
                                                 cNames=["compCon0","compCon1","compCon2"])

    # Define Artelys Knitro non default options
    options = {}
    options['outlev'] = KN_OUTLEV_ALL

    # Solve the problem
    solution=optimize(variables=variables,
                      objective=objective,
                      constraints=constraints,
                      compConstraints=compConstraints,
                      options=options)


### Single call function NLP1 example
def example_single_call_NLP1():
    # ExampleNLP1
    # min   100 (x1 - x0^2)^2 + (1 - x0)^2
    # s.t.  x0 x1 >= 1
    #       x0 + x1^2 >= 0
    #       x0 <= 0.5

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalF                                       *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "obj" is set in the KN_eval_result structure.
    def eval_f(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        # Evaluate nonlinear objective
        dTmp = x[1] - x[0]*x[0]
        evalResult.obj = 100.0 * (dTmp*dTmp) + ((1.0 - x[0]) * (1.0 - x[0]))
        return 0

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalG                                       *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "objGrad" is set in the KN_eval_result structure.
    def callbackEvalG(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
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
    def callbackEvalH(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
            print("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
            print(evalRequest.type)
            return -1
        x = evalRequest.x
        sigma = evalRequest.sigma
        evalResult.hess[0] = sigma * ((-400.0 * x[1]) + (1200.0 * x[0]*x[0]) + 2.0) # (0,0)
        evalResult.hess[1] = sigma * (-400.0 * x[0]) # (0,1)
        evalResult.hess[2] = sigma * 200.0           # (1,1)
        return 0

    # Define the variables information
    variables = Variables(nV=2,
                          xLoBnds=[-KN_INFINITY, -KN_INFINITY], # not necessary since infinite
                          xUpBnds=[0.5, KN_INFINITY],
                          xInitVals=[-2.0, 1.0])

    # Define the constraints information
    constraints = Constraints(nC=2, cLoBnds=[1.0, 0.0],
                              cLinear=[[1],[0],[1.0]],
                              cQuadratic=[[0,1],[0,1],[1,1],[1.0,1.0]])

    # Define Artelys Knitro non default options
    # i.e. check user provided derivatives.
    options = {}
    options['derivcheck']   = KN_DERIVCHECK_ALL

    # Define general non linear objective terms and constraints.
    # Provide callbacks to evaluate the corresponding functions and derivatives.
    callback = Callback(evalObj=True,
                        funcCallback=eval_f,
                        objGradIndexVars=KN_DENSE,
                        gradCallback=callbackEvalG,
                        hessIndexVars1=KN_DENSE_ROWMAJOR,
                        hessCallback=callbackEvalH,
                        hessianNoFAllow=True)

    # Solve the problem
    solution=optimize(variables=variables,
                      constraints=constraints,
                      callbacks=callback,
                      options=options)


### Single call function NLP1NoDerivs example
def example_single_call_NLP1NoDerivs():
    # ExampleNLP1
    # min   100 (x1 - x0^2)^2 + (1 - x0)^2
    # s.t.  x0 x1 >= 1
    #       x0 + x1^2 >= 0
    #       x0 <= 0.5

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalF                                       *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "obj" is set in the KN_eval_result structure.
    def eval_f(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        # Evaluate nonlinear objective
        dTmp = x[1] - x[0]*x[0]
        evalResult.obj = 100.0 * (dTmp*dTmp) + ((1.0 - x[0]) * (1.0 - x[0]))
        return 0

    # Define the variables information
    variables = Variables(nV=2,
                          xLoBnds=[-KN_INFINITY, -KN_INFINITY], # not necessary since infinite
                          xUpBnds=[0.5, KN_INFINITY],
                          xInitVals=[-2.0, 1.0])

    # Define the constraints information
    constraints = Constraints(nC=2, cLoBnds=[1.0, 0.0],
                              cLinear=[[1],[0],[1.0]],
                              cQuadratic=[[0,1],[0,1],[1,1],[1.0,1.0]])

    # Define general non linear objective terms and constraints.
    # Provide callbacks to evaluate the corresponding functions.
    # Derivatives will be approximated by the solver.
    callback = Callback(evalObj=True,
                        funcCallback=eval_f)

    # Solve the problem
    solution=optimize(variables=variables,
                      constraints=constraints,
                      callbacks=callback)


### Single call function FCGA example
def example_single_call_FCGA():
    # ExampleFCGA
    #  max   x0*x1*x2*x3         (obj)
    #  s.t.  x0^3 + x1^2 = 1     (c0)
    #        x0^2*x3 - x2 = 0    (c1)
    #        x3^2 - x1 = 0       (c2)

    #*------------------------------------------------------------------*/
    #*     FUNCTION callbackEvalFCGA                                    */
    #*------------------------------------------------------------------*/
    # The signature of this function matches KN_eval_callback in knitro.h.
    # To compute the functions and gradient together, set "obj", "c",
    # "objGrad" and "jac" in the KN_eval_result structure when there
    # is a function+gradient evaluation request (i.e. EVALFCGA).
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

    # Define the variables information
    variables = Variables(nV=4, xInitVals=[0.8]*4)

    # Define the constraints information
    constraints = Constraints(nC=3,
                      cEqBnds=[1.0, 0.0, 0.0],
                      cLinear=[[1, 2],[2, 1],[-1.0, -1.0]],
                      cQuadratic=[[0,2],[1,3],[1,3],[1.0,1.0]])

    # Define the objective information
    objective = Objective(objGoal=KN_OBJGOAL_MAXIMIZE)

    # Define general non linear objective terms and constraints.
    # Use of evalFCGA to specify that first derivatives/gradients are
    # also evaluated in the funcCallBack. In that case, gradCallback
    # is not required. In this example, no callback is provided for
    # the Hessian of the Lagrangian, it is automatically approximated.
    callback = Callback(evalObj=True,
                        evalFCGA=True,
                        indexCons=[0,1],
                        funcCallback=callbackEvalFCGA,
                        jacIndexCons=[0, 1, 1],
                        jacIndexVars=[0, 0, 3],
                        objGradIndexVars=KN_DENSE)

    # Define Artelys Knitro non default options
    options = {}
    options['derivcheck'] = KN_DERIVCHECK_ALL
    options['outlev']     = KN_OUTLEV_ITER

    # Solve the problem
    solution=optimize(variables=variables,
                      objective=objective,
                      constraints=constraints,
                      callbacks=callback,
                      options=options)


### Single call function MINLP1 example
def example_single_call_MINLP1():
    # min   5 x3 + 6 x4 + 8 x5 + 10 x0 - 7 x2 -18 log(x1 + 1)
    #      - 19.2 log(x0 - x1 + 1) + 10
    # s.t.  0.8 log(x1 + 1) + 0.96 log(x0 - x1 + 1) - 0.8 x2 >= 0
    #       log(x1 + 1) + 1.2 log(x0 - x1 + 1) - x2 - 2 x5 >= -2
    #       x1 - x0 <= 0
    #       x1 - 2 x3 <= 0
    #       x0 - x1 - 2 x4 <= 0
    #       x3 + x4 <= 1
    #       0 <= x0 <= 2
    #       0 <= x1 <= 2
    #       0 <= x2 <= 1
    #       x0, x1, x2 continuous
    #       x3, x4, x5 binary
    #
    # The solution is (1.30098, 0, 1, 0, 1, 0).

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalFC                                      *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "obj" and "c" are set in the KN_eval_result structure.
    def callbackEvalFC (kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalFC incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x

        # Evaluate nonlinear objective structure
        dTmp1 = x[0] - x[1] + 1.0
        dTmp2 = x[1] + 1.0
        evalResult.obj = -18.0*math.log(dTmp2) - 19.2*math.log(dTmp1)

        # Evaluate nonlinear constraint structure
        evalResult.c[0] = 0.8*math.log(dTmp2) + 0.96*math.log(dTmp1)
        evalResult.c[1] = math.log(dTmp2) + 1.2*math.log(dTmp1)

        return 0

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalGA                                      *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "objGrad" and "jac" are set in the KN_eval_result structure.
    def callbackEvalGA (kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalGA incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x

        # Evaluate gradient of nonlinear objective structure
        dTmp1 = x[0] - x[1] + 1.0
        dTmp2 = x[1] + 1.0
        evalResult.objGrad[0] = -(19.2 / dTmp1)
        evalResult.objGrad[1] = (-18.0 / dTmp2) + (19.2 / dTmp1)

        # Gradient of nonlinear structure in constraint 0.
        evalResult.jac[0] = 0.96 / dTmp1                      # wrt x0
        evalResult.jac[1] = (-0.96 / dTmp1) + (0.8 / dTmp2)   # wrt x1
        # Gradient of nonlinear structure in constraint 1.
        evalResult.jac[2] = 1.2 / dTmp1                       # wrt x0
        evalResult.jac[3] = (-1.2 / dTmp1) + (1.0 / dTmp2)    # wrt x1

        return 0

    #*------------------------------------------------------------------*
    #*     FUNCTION callbackEvalH                                       *
    #*------------------------------------------------------------------*
    # The signature of this function matches KN_eval_callback in knitro.py.
    # Only "hess" or "hessVec" are set in the KN_eval_result structure.
    def callbackEvalH (kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
            print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
            return -1
        x = evalRequest.x
        lambda_ = evalRequest.lambda_
        # Scale objective component of hessian by sigma
        sigma = evalRequest.sigma

        # Evaluate the non-zero components in the Hessian of the Lagrangian.
        # Note: Since the Hessian is symmetric, we only provide the
        #       nonzero elements in the upper triangle (plus diagonal).
        dTmp1 = x[0] - x[1] + 1.0
        dTmp2 = x[1] + 1.0
        evalResult.hess[0] = sigma*(19.2 / (dTmp1*dTmp1))          # (0,0)
        + lambda_[0]*(-0.96 / (dTmp1*dTmp1))
        + lambda_[1]*(-1.2 / (dTmp1*dTmp1))
        evalResult.hess[1] = sigma*(-19.2 / (dTmp1*dTmp1))         # (0,1)
        + lambda_[0]*(0.96 / (dTmp1*dTmp1))
        + lambda_[1]*(1.2 / (dTmp1*dTmp1))
        evalResult.hess[2] = sigma*((19.2 / (dTmp1*dTmp1)) + (18.0 / (dTmp2*dTmp2))) # (1,1)
        + lambda_[0]*((-0.96 / (dTmp1*dTmp1)) - (0.8 / (dTmp2*dTmp2)))
        + lambda_[1]*((-1.2 / (dTmp1*dTmp1)) - (1.0 / (dTmp2*dTmp2)))

        return 0

    # Define the variables information
    variables = Variables(nV=6,
                          xLoBnds=[0]*6,
                          xUpBnds=[2, 2, 1, 1, 1, 1],
                          xInitVals=[0.8]*4,
                          xTypes=[KN_VARTYPE_CONTINUOUS,
                              KN_VARTYPE_CONTINUOUS,
                              KN_VARTYPE_CONTINUOUS,
                              KN_VARTYPE_BINARY,
                              KN_VARTYPE_BINARY,
                              KN_VARTYPE_BINARY],
                          xProperties=[[2,3,4,5],[KN_VAR_LINEAR]*4],
                          xNames=["var0", "var1", "var2", "var3","var4", "var5"])

    # Define the constraints information
    constraints = Constraints(nC=6,
                              cLoBnds=[[0,1], [0, -2]],
                              cUpBnds=[KN_INFINITY, KN_INFINITY, 0, 0, 0, 1],
                              cLinear=[[0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
                                       [2, 2, 5, 1, 0, 1, 3, 0, 1, 4, 3, 4],
                                       [-0.8, -1, -2, 1, -1, 1, -2, 1, -1, -2, 1, 1]])

    # Define the objective information
    objective = Objective(objGoal='minimize',
                          objConstant=10.0,
                          objLinear=[[3, 4, 5, 0, 2], [5.0, 6.0, 8.0, 10.0, -7.0]])

    # Define general non linear objective and constraints terms.
    callbacks = Callback(evalObj=True,
                         hessianNoFAllow=True,
                         indexCons=[0,1],
                         funcCallback=callbackEvalFC,
                         jacIndexCons=[0, 0, 1, 1],
                         jacIndexVars=[0, 1, 0, 1],
                         gradCallback=callbackEvalGA,
                         objGradIndexVars=[0, 1],
                         hessIndexVars1=[0, 0, 1],
                         hessIndexVars2=[0, 1, 1],
                         hessCallback=callbackEvalH)

    # Define Artelys Knitro non default options
    options = {}
    options['mip_outinterval'] = 1
    options['mip_maxnodes']    = 10000
    options['mip_method']      = KN_MIP_METHOD_BB
    options['algorithm']       = KN_ALG_ACT_CG
    options['outmode']         = KN_OUTMODE_SCREEN
    options['outlev']          = KN_OUTLEV_ALL

    # Solve the problem
    solution=optimize(variables=variables,
                      objective=objective,
                      constraints=constraints,
                      callbacks=callbacks,
                      options=options)

example_single_call_LP1()
example_single_call_MPEC1()
example_single_call_NLP1()
example_single_call_NLP1NoDerivs()
example_single_call_FCGA()
example_single_call_MINLP1()
