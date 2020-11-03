#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 10:11:44 2019

@author: dipankarbhattacharya
"""

#!python
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import time
# from knitro import *
import pdb

from sindybase import *
from tvregdiff_master.tvregdiff import TVRegDiff

plt.close('all')

def cls():
    print("\n" * 50)


# clear Console
cls()

class SINDYc_MPC_Design(SINDyBase):
    
    def __init__(self, SystemName, tspan):
#        self.path = path
        self.SystemName = SystemName
#        self.name = name
#        self.currWorkDir = os.getcwd()
#        os.chdir(
#            self.path
#        )  # whenever we will create an instance of the class, the path will be set for it
#        self.DataFrameExtractedDict = {}
#        self.ArrayDict = {}
#        self.MeanMigrationInfoDict = {}
#        self.GradMeanMigrationInfoDict = {}
        self.DataDictionary={}
        self.tspan=tspan
        self.n_tspan=self.tspan.shape[0]
        self.x=np.zeros((self.n_tspan,2))
    
    #Lotka-Voltera System with input    
    @staticmethod
    def lotkacontrol(y,t,a,b,c,d,u):
        """ Return the growth rate of fox and rabbit populations. """
        dy=np.array([ a*y[0] -   b*y[0]*y[1] ,
                      -c*y[1] + d*y[0]*y[1] +u])
        return dy
    
    @staticmethod
    def chirp(tspan, start_Hz, stop_Hz, phase_rad = 0):
        """
        #chirp
        #
        # Generate a frequency sweep from low to high over time.
        # Waveform description is based on number of samples.
        #
        # Inputs
        #  numSamples: integer, number of samples in chirp waveform.
        #  chirpLen_s: float, time span in seconds of chirp.
        #  start_Hz: float, start (lower) frequency in Hz of chirp.
        #  stop_Hz: float, stop (upper) frequency in Hz of chirp.
        #  phase_rad: phase in radians at waveform start, default is 0.
        #
        # Output
        #  Time domain chirp waveform of length numSamples.
        """      
    
        times_s = tspan # Chirp times.
        chirpLen_s=tspan.shape[0]-1
        k = (stop_Hz - start_Hz) / chirpLen_s # Chirp rate.
        sweepFreqs_Hz = (start_Hz + k/2. * times_s) * times_s
        chirp = np.sin(phase_rad + 2 * np.pi * sweepFreqs_Hz)
        
        return chirp
    
    @staticmethod
    def GenerateInput(tspan,InputSignalType='chirp'):
#        pdb.set_trace()
        
        if InputSignalType=='sine2':
            
            A=2
            u=[(A*mt.sin(1*t)+A*mt.sin(0.1*t))**2 for t in tspan]
            u=np.array(u)
            
            return u
        
        elif InputSignalType=='chirp':
            
            A=2
            u=A-A*SINDYc_MPC_Design.chirp(tspan,0,1)**2;
            return u
    
  
    def SplitData2TrainAndValid(self,u,x,split=0.8,selection='defined'):
        """
        :param mtx: the theta matrix of shape (M, N)
        :param _b: a vector or an array of shape (M,) or (M, K)
        :param init_tol: maximum tolerance (cut_off value)
        :param max_iter: maximum iteration of the outer loop
        :param thresh_iter: maximum iteration for threshold least squares
        :param l0_penalty: penalty factor for nonzero coefficients
        :param split: proportion of the training set
        :param normalize: normalization methods, default as 0 (no normalization)
        :return: the best coefficients of fit
        """
#        pdb.set_trace()
        if x==[]:
            x=self.x

        if u.ndim == 1:
            u = u[:, np.newaxis]
            
        
        if selection=='defined':
            
            _n = x.shape[0]
            train = np.arange(int(_n*split))
            valid = np.arange(int(_n*split),_n)
            x_train = x[train, :]
            x_valid =  x[valid, :]
            u_train = u[train, :]
            u_valid = u[valid, :]
            
            
        elif selection=='random':

            # split the data
            np.random.seed(12345)
            _n = x.shape[0]
            train = np.random.choice(_n, int(_n*split), replace=False)
            valid = [x for x in np.arange(_n) if x not in train]
            x_train = x[train, :]
            x_valid =  x[valid, :]
            u_train = u[train, :]
            u_valid = u[valid, :]
        
        self.DataDictionary={'x_train':x_train,
                             'x_valid':x_valid,
                             'u_train':u_train,
                             'u_valid':u_valid
                             }
    
    #Simulation of Lotka-Voltera System with input 
    def ODEINTSimulation(self,system,x0,u,tspan,*DEParameters):
#        pdb.set_trace()
        
        self.x=np.zeros((u.shape[0],x0.shape[0]))
        
        print(u.shape)
        self.x[0,:]=x0
        
        for i in range(1,u.shape[0]):
            t=[tspan[i-1],tspan[i]]
            y_temp= odeint(system, x0, t,args=DEParameters[0]+(u[i],))
            x0=y_temp[1,:]
            self.x[i,:]=y_temp[1,:]
            
        self.tspan=tspan
        
        return tspan, self.x
        
#       
    def SparseGalerkinControl(self,y,t,Xi,degree,u):

#        pdb.set_trace()
        
        Xi=Xi[:,0:2]
        
        y=np.array(y,ndmin=2,dtype=float)
        
        y_u=np.vstack((y.T,u)).T
#        y_u1=np.array(x_u[0,:],ndmin=2)
        ypool,_=self.polynomial_expansion(y_u,degree)
#        dy=ypool.dot(Xi)
        dy=np.matmul(ypool,Xi)
        dy=dy[0,0:2]
         
        return dy
   
    @staticmethod
    def ComputeDerivative(DiffrentiationType,x,u,dt,noise=False,eps=0):
        
#        pdb.set_trace()
            
        if eps==0:
            DERIV_NOISE=0 
            
        if u.ndim == 1:
            u = u[:, np.newaxis]
                    
        Shape_x_train=x.shape 
        
        if noise:
        # creating a noise with the same dimension as the dataset
#            pdb.set_trace()
            mu, sigma = 0, 0.1 
            noise = np.random.randn(Shape_x_train[0],Shape_x_train[1]) #np.random.normal(mu, sigma, [Shape_x_train[0],2] )
            
            x=x+eps*noise
        
        if DiffrentiationType=='CentralDiff':
            
            #Compute derivative using fourth order central difference
            #Use TVRegDiff if more error
            
            dx=np.zeros((Shape_x_train[0]-5,3))
            
            dx[:,0:2]=np.array([(1/(12*dt))*(-x[ii+2,kk]+8*x[ii+1,kk]-8*x[ii-1,kk]+x[ii-2,kk]) \
                         for ii in range(2,Shape_x_train[0]-3) \
                         for kk in range(Shape_x_train[1])]).reshape((dx.shape[0],2))
                        
            x_aug=np.concatenate((x[3:-2,:],u[3:-2,:]),axis=1)
            
        elif DiffrentiationType=='TVRegDiff':
            
            dx=np.zeros((Shape_x_train[0],3))
#            pdb.set_trace()
            #total Variation Regularized Differentiation
            for ii in range(Shape_x_train[1]):
                dx[:,ii]=TVRegDiff(x[:,ii],20,0.00002,u0=None, scale='small', ep=1e12, dx=dt,plotflag=False)[1:]
                
            #Recovering the data from noisy data by cummulative summation (Integration in discrete)
            xt=np.cumsum(dx[:,0:2],axis=0)*dt
            xt = xt- (np.mean(xt[1000:-1000],axis=0) - np.mean(x[1000:-1000],axis=0))
            
            xt = xt[500:-500]
            dx = dx[500:-500]  #trim off ends (overly conservative)
            x_aug=np.concatenate((xt,u[500:-500]),axis=1)  

        return dx,x_aug
    
    def TrainSINDYc(self,degree,x_aug=[],dx=[],lam_bda=1e-3,usesine=False,Normalization=False):
#      pdb.set_trace() 
      self.SINDYcTrainParam={'degree':degree,
                                 'lambda':lam_bda,
                                 'usesine':usesine}
      if x_aug==[] and dx==[]:

          
          if Normalization:
          #Compute Theta
              Theta,Theta_desp=self.polynomial_expansion(self.x_aug,degree)  
              #Perform normalization on Theta
              Theta_norm= np.sum(Theta**2,axis=0)[:,np.newaxis].T**0.5
              Theta/=Theta_norm
              
              # Compute Sparse coeefecients
              Xi, _ =self.sparsify_dynamics(Theta, self.dx, lam_bda,split=0.99)
              Xi/=Theta_norm.T
              
              self.Theta_norm=Theta_norm
              
          else:
            #Compute Theta
              Theta,Theta_desp=self.polynomial_expansion(self.x_aug,degree)  
    
              # Compute Sparse coeefecients
              #Xi, _ =self.sparsify_dynamics(Theta, self.dx, lam_bda,split=0.99)
              Xi=SINDYc_MPC_Design.threshold_ls(Theta, dx, lam_bda)
          
      else:
          
          Theta,Theta_desp=self.polynomial_expansion(x_aug,degree) 
          #Xi, _ =self.sparsify_dynamics(Theta, dx, lam_bda,split=0.99)
          Xi=SINDYc_MPC_Design.threshold_ls(Theta, dx, lam_bda)
                    
      self.Theta=Theta
      self.Theta_desp=Theta
      self.Xi=Xi
      
      return Theta,Theta_desp,Xi
  
    # =============================================================================
    #     Runge-Kutta Method
    # =============================================================================
    
    def Rk4u(self,fun_discrete,x,u,h,n,t,p):
        """
        # RK4U   Runge-Kutta scheme of order 4 for control system
        #   rk4u(v,X,U,h,n) performs n steps of the scheme for the vector field v
        #   using stepsize h on each row of the matrix X
        #
        #   v(X,U) maps an (m x d)-matrix X and an (m x p)-matrix U
        #          to an (m x d)-matrix 
        """
#        pdb.set_trace()

        
        for ii in range(n):
            k1 = k1=fun_discrete(x,t,u,p).astype('float64')
            k2 = k2=fun_discrete(x + h/2*k1,t,u,p).astype('float64')
            k3 = k3=fun_discrete(x + h/2*k2,t,u,p).astype('float64')
            k4 = k4=fun_discrete(x + h*k3,t,u,p).astype('float64')
            x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
            
        return x
    
    def lotkacontrol_discrete(self,y,t,u,p):
        
        a=p.a
        b=p.b
        c=p.c
        d=p.d
        
        """ Return the growth rate of fox and rabbit populations. """
        
        dy=np.array([ a*y[0] -   b*y[0]*y[1] ,
                      -c*y[1] + d*y[0]*y[1] +u])
        
        return dy
    
    def SparseGalerkinControl_Discrete(self,y,t,u,p):

#        pdb.set_trace()
        if hasattr(p, 'SelectVars') is False:
            p.SelectVars=np.arange(x.shape[0])
#            p.SelectVars=p.SelectVars[:,np.newaxis]
            
        Xi=p.Xi
        degree = p.polyOrder;
        usesine = p.usesine;
        
        y=np.array(y,ndmin=2,dtype=float)
        
        y_u=np.vstack((y[:,p.SelectVars].T,u)).T
#        y_u1=np.array(x_u[0,:],ndmin=2)
        ypool,_=self.polynomial_expansion(y_u,degree)
#        dy=ypool.dot(Xi)
        dy=np.matmul(ypool,Xi)
        dy=dy[0,0:2]
         
        return dy
# =============================================================================
#     Cost function of nonlinear MPC for Lotka-Volterra system
# =============================================================================
    def lotkaObjectiveFCN(self,u,x,Ts,N,xref,u0,p,Q,R,Ru):
    
    ## 
        """
         Inputs:
           u:      optimization variable, from time k to time k+N-1 
           x:      current state at time k
           Ts:     controller sample time
           N:      prediction horizon
           xref:   state references, constant from time k+1 to k+N
           u0:     previous controller output at time k-1
        
         Output:
           J:      objective function cost
        """
    
    # Cost calculation
#        pdb.set_trace()
        xk=x
        uk=u[0]
        J=0
        
    #    Loop through each prediction step
        for kk in range(N):
    #        Obtain plant state at next prediction step
            xk1 = self.Rk4u(self.SparseGalerkinControl_Discrete,
                                         xk,uk,Ts,1,[],p)
            
            #J+=np.matmul(np.matmul((xk1-xref).T,Q),(xk1-xref))
            J+=(xk1-xref).T.dot(Q).dot((xk1-xref))
            
            if kk==0:
                J+=np.dot(np.dot((uk-u0).T,R),(uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
                
#                (uk-u0).T.dot(R).dot((uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
            else:
                J+=np.dot(np.dot((uk-u[kk-1]).T,R),(uk-u[kk-1]))+np.dot(np.dot(uk.T,Ru),uk)
                
            # Update xk and uk for the next prediction step
            
            xk=xk1
            
            if kk<N-1:
                uk=u[kk+1]
                
        return J

# =============================================================================
# Constraint function of nonlinear MPC for Lotka-Volterra system
# =============================================================================
        
    def lotkaConstraintFCN(self,u):
        """
        % Inputs:
        %   u:      optimization variable, from time k to time k+N-1 
        %   x:      current state at time k
        %   Ts:     controller sample time
        %   N:      prediction horizon
        %
        % Output:
        %   c:      inequality constraints applied across prediction horizon
        %   ceq:    equality constraints (empty)
        """
        # Nonlinear MPC design parameters
        #Predator population size always positive: >0, min pop size
        x,Ts,N,p=args_userdef
        
        zMin=100
#        pdb.set_trace()
        
        #Inequality constraints calculation
        
        c=np.zeros(N)
        
        #Apply N population size constraints across prediction horizon, from time
        #k+1 to k+N
        
        xk = x
        uk = u[0]
        
        for kk in range(N):
            #obtain new cart position at next prediction step
           
            xk1 = self.Rk4u(self.SparseGalerkinControl_Discrete,
                                         xk,uk,Ts,1,[],p)
            
            # -z + zMin < 0 % lower bound
            c[kk] = -xk1[1]+zMin #c(2*ct-1)
            
            #z - zMax < 0 % upper bound
            #c(2*ct) = xk1(1)-zMax;
            #update plant state and input for next step
            
            xk=xk1
            
            if kk<N-1:
                uk=u[kk+1]
                
            #No equality constraints
            ceq=[]
            
            # print(c)
                
        return c

# =============================================================================
# Generate input data    
# =============================================================================

InputSignalType='sine2'
ONLY_TRAINING_LENGTH = 1;
DERIV_NOISE = 0;

# Definition of parameters
a = 0.5
b = 0.025
c = 0.5
d = 0.005
DEParameters=(a,b,c,d)

x0=np.array([60.0, 50.0])
xref = np.array([c/d,a/b]) # critical point

dt=0.01

tspan=np.linspace(0, 200, 20001)

LotkaVolteraSysObj_SINDYc_SI=SINDYc_MPC_Design('LotkaVoltera_SINDYc_SI',tspan)

u=SINDYc_MPC_Design.GenerateInput(LotkaVolteraSysObj_SINDYc_SI.tspan,InputSignalType)

# =============================================================================
# Genrate Training Data by simulating the Lotka-Voltera equations
# =============================================================================
tspan_simu,x_Lot_Vol_simu=LotkaVolteraSysObj_SINDYc_SI.ODEINTSimulation(LotkaVolteraSysObj_SINDYc_SI.lotkacontrol,
                                              x0,u,
                                              LotkaVolteraSysObj_SINDYc_SI.tspan,
                                              DEParameters)
x_Lot_Vol_simu_clean=x_Lot_Vol_simu
LotkaVolteraSysObj_SINDYc_SI.SplitData2TrainAndValid(u,[],split=0.5)
x=LotkaVolteraSysObj_SINDYc_SI.DataDictionary['x_train']
u=u[0:x.shape[0],np.newaxis]
# =============================================================================
# Compute derivative
# =============================================================================
xold = x;
eps = 0.1;
dx=np.zeros((x.shape[0],x.shape[1]+1))
for i in range(x.shape[0]):
        dx[i,0:2] = SINDYc_MPC_Design.lotkacontrol(x[i,:],0,a,b,c,d,u[i]);
        
x_u = np.concatenate((xold,u),axis=1)
dxClean=dx
dxNoise = dxClean + eps*np.random.randn(dxClean.shape[0],dxClean.shape[1]);

# =============================================================================
# Apply Sparse regression on noisy data
# =============================================================================

#Sparse Regression
polyOrder=3
_,_,XiNoise=LotkaVolteraSysObj_SINDYc_SI.TrainSINDYc(degree=polyOrder,
                                         x_aug=x_u,
                                         dx=dxNoise ,
                                         lam_bda=0.001)

# =============================================================================
# Prediction over clean data and Actual DE system
# =============================================================================
x0_pred_actual=x[-1,0:2]
tspan_pred=tspan[10000:-1]
u_tspan_pred=SINDYc_MPC_Design.GenerateInput(tspan_pred,InputSignalType)

tspan_pred,x_Lot_Vol_pred=LotkaVolteraSysObj_SINDYc_SI.ODEINTSimulation(LotkaVolteraSysObj_SINDYc_SI.lotkacontrol,
                                              x0_pred_actual,
                                              u_tspan_pred,
                                              tspan_pred,
                                              DEParameters)

## =============================================================================
## Prediction over noisy data and Sparse regression system
## =============================================================================
#degree=polyOrder
#DEParameters_SINDYc_noisy=(XiNoise,degree)
#x0_pred_noisy=x[-1,0:2]
#
#tspan_pred_noisy,x_pred_noisy=LotkaVolteraSysObj_SINDYc_SI.ODEINTSimulation(LotkaVolteraSysObj_SINDYc_SI.SparseGalerkinControl,
#                                               x0_pred_noisy,
#                                               u_tspan_pred,
#                                               tspan_pred,
#                                               DEParameters_SINDYc_noisy)

# =============================================================================
# Apply Model predictive controlto system using SINDYc model
# =============================================================================
x0=np.array([69.99,30.59])

class MPCParameters:
    pass
pest=MPCParameters
pest.Xi=XiNoise[:,0:2]
pest.polyOrder = polyOrder
pest.usesine = False

# True model parameters
class MPCParameters2:
    pass
p=MPCParameters2
p.a=a
p.b=b
p.c=c
p.d=d

# Choose prediction horizon over which the optimization is performed
Nvec=np.array([3])

# =============================================================================
# 
# =============================================================================

for i in range(Nvec.shape[0]):
    
    Ts          = 0.1              # Sampling time
    N           = Nvec[i]          # Control / prediction horizon (number of iterations)
    Duration    = 100              # Run control for 100 time units
    Nvar        = 2
    Q           = np.array([1,0])            # State weights
    #Q=Q[np.newaxis,:]
    R           = 0.5 #0.5;         # Control variation du weights
    Ru = 0.5#0.01                 # Control weights
#    B = np.array([0,1])                    # Control vector (which state is controlled)
#    B=B[:,np.newaxis]
#    #C = eye(Nvar)                  # Measurement matrix
#    D = 0                          # Feedforward (none)
    x0n=x0.T#[100; 50];             # Initial condition
    uopt0 = 0                      # Set initial control input to zero
    
    # Constraints on control optimization
    LB = [] #-100*ones(N,1);        # Lower bound of control input
    UB = [] #100*ones(N,1);         # Upper bound of control input
       
    # Reference state, which shall be achieved
    xref1 = np.array([c/d,a/b]).T # critical point
    
    
    x        = x0n
    x=np.asarray(x)
    
    Ton      = 30      # Time when control starts
    uopt     = uopt0*np.ones(N)
    uopt=np.asarray(uopt)
    
    xHistory = x       # Stores state history
    uHistory = uopt[0] # Stores control history
    tHistory = np.zeros((1))       # Stores time history
    rHistory = xref1   # Stores reference (could be trajectory and vary with time)
    
    #pdb.set_trace()
    
    bound_optVar=[(-np.Inf,np.Inf)]
    
    funval=np.zeros((int(Duration/Ts)))
    
    start_time = time.time()
       
    for ct in range(int(Duration/Ts)):
        
        if ct*Ts>30:   # Turn control on
            
            if ct*Ts==Ton+Ts:
                print('Start Control.')
        
            # Set references
            xref = np.asarray(xref1)
            
            obj=LotkaVolteraSysObj_SINDYc_SI
        
            #% NMPC with full-state feedback    
            args_userdef=(x,Ts,N,pest)
            nonlinear_constraint=NonlinearConstraint(obj.lotkaConstraintFCN,
                                                     -np.Inf,
                                                     np.Inf)
            
            
                        
        #    pdb.set_trace()
            res=minimize(obj.lotkaObjectiveFCN,
                          uopt,
                          method='trust-constr',
                          args=(x,Ts,N,xref,uopt[0],pest,np.diag(Q),R,Ru),
                          options={'verbose': 0,'maxiter':2,'xtol':1e-01,'gtol':1e-01},
                          constraints=[],
                          # bounds=bound_optVar
                          )
#            pdb.set_trace()
            
            
            funval[ct]=res.fun
            
            if np.absolute(x[0])>700 or np.absolute(x[1])>700:
                break
            
            uopt=res.x
            
        else:    #% If control is off
            
            uopt=uopt0*np.ones((N))
            xref=-1000*np.ones((2))
            
        # Integrate system: Apply control & Step one timestep forward
            
        x = LotkaVolteraSysObj_SINDYc_SI.Rk4u(LotkaVolteraSysObj_SINDYc_SI.lotkacontrol_discrete,x,uopt[0],Ts/1,1,[],p)
        
        
        xHistory=np.vstack((xHistory,x))
        uHistory=np.vstack((uHistory,uopt[0]))
        tHistory = np.vstack((tHistory,tHistory[-1]+Ts/1))
        rHistory = np.vstack((rHistory,xref))
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
    # =============================================================================
    # Create a Dicionary for storing the data history
    # =============================================================================
    HistoryDict={'xHistory':xHistory,
                 'uHistory':uHistory,
                 'tHistory':tHistory,
                 'rHistory':rHistory,
                 'JHistory':funval}
    
    # =============================================================================
    #     Show results
    # =============================================================================
    plt.close('all')
    
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
     
    ax1.plot(np.array([Ton+tspan[0],Ton+tspan[0]]),np.array([15,260]),
             color='limegreen',
                alpha=0.8,
                linestyle='--',
                linewidth=1.5)
    
    ax1.plot(tHistory+tspan[0],np.zeros((tHistory.shape[0])),
            color='k',
            linewidth=1.5,
            linestyle='--')
    
    ax1.plot(tHistory+tspan[0],xref1[0]*np.ones((tHistory.shape[0])),
            color='skyblue',
            linewidth=1.5,
            linestyle='--')
    
    ax1.plot(tHistory+tspan[0],xref1[1]*np.ones((tHistory.shape[0])),
            color='darkorange',
            linewidth=1.5,
            linestyle='--')
    
    ph0 = ax1.plot(tHistory+tspan[0],xHistory[:,0],
               color='darkblue',
               linewidth=1.5,
               linestyle='-',
               label='Prey')
    
    ph1 = ax1.plot(tHistory+tspan[0],xHistory[:,1],
               color='darkred',
               linewidth=1.5,
               linestyle='-',
               label='Predator')
    
    ph2 = ax1.plot(tHistory+tspan[0],uHistory[:,0],
               color='brown',
               linewidth=1.5,
               linestyle='-',
               label='Control')
    
    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    
    ax1.set_xlabel(r"Time", size=12)
    ax1.set_ylabel(r"Population Size", size=12)
    plt.show()
# =============================================================================
# Objective function
# =============================================================================
    f2 = plt.figure()
    ax1 = f2.add_subplot(111)
     
    ax1.plot(tHistory[1:,:]+tspan[0],funval,
               color='darkblue',
               linewidth=1.5,
               linestyle='-',
               label='Obejcetive function')
    
    ax1.set_xlabel(r"Time", size=12)
    ax1.set_ylabel(r"J", size=12)
    plt.show()
    
# =============================================================================
# Control Vector
# =============================================================================
    f3 = plt.figure()
    ax1 = f3.add_subplot(111)
    
    ph2 = ax1.plot(tHistory+tspan[0],uHistory[:,0],
                   color='brown',
                   linewidth=1.5,
                   linestyle='-',
                   label='Control')
    
    ax1.set_xlabel(r"Time", size=12)
    ax1.set_ylabel(r"u(t)", size=12)
    plt.show()