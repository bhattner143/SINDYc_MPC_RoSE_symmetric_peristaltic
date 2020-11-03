# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:43:57 2020

@author: useradmin-dbha483
"""

import os
import pdb
import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import pandas as pd
import random
# from intelhex import IntelHex
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.nonparametric.smoothers_lowess import lowess
from dateutil.parser import parse
from scipy.signal import find_peaks, peak_prominences, kaiserord, lfilter, firwin, freqz
from scipy import zeros, signal, random
from matplotlib import style
from tvregdiff_master.tvregdiff import TVRegDiff
from scipy.integrate import odeint
from sindybase import *
from ClassTOFandWebCam import *



#from Class_SINDYc_MPC_Design import *


def cls():
    print("\n" * 50)


# clear Console
cls()

# =============================================================================
# MPC design Class
# =============================================================================
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
        if x.size==0:
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
            
            dx=np.zeros((Shape_x_train[0],1))
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
        
    def lotkaConstraintFCN(self,u,x,Ts,N,p):
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
# MAIN PROGRAM
# =============================================================================
# Plotting
# plt.rcParams['figure.figsize'] = (16.0, 10) # set default size of plots #4.2x1.8

plt.style.use("classic")
# sns.set_style("white")
# sns.set_style("dark")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 6
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["image.interpolation"] = "nearest"
# plt.rcParams['font.family']='Helvetica'
plt.rcParams["lines.markersize"] = 3
plt.rc("lines", mew=0.5)
plt.rcParams["lines.linewidth"] = 1
#matplotlib.rcParams.update({"errorbar.capsize": 1})
plt.close("all")


# =============================================================================
# Make dict of TOF, ADC, and Peristalsis data with time
# =============================================================================
GenPath = os.getcwd()
path = GenPath+"/DataFiles/Data29_06_2020/"
name = "TOFADCandPer"
Bolustype = "Dry"
QSR_DoubleLayer_TOFADCandPer = TOFandWebCam(path, "TOFADCandPer_QSR_40mmat20mmps", name)
QSR_DoubleLayer_TOFADCandPer.find_all()
QSR_DoubleLayer_TOFADCandPer.FileReading([4])

dict_keys=list(QSR_DoubleLayer_TOFADCandPer.ArrayDict.keys())

# t_data={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][:,0]}
t_data={'1.0':np.linspace(0,QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,0].shape[0],\
            QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,0].shape[0]+1)}
u_data={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,[20]]}
x_adc={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,[8]]}
x_tof_raw={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,[2]]}
x_tof_online_fil={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][100:,[4]]}



#Design filter
sample_rate = 100.0
tr_band=4
ripple_db = 40.0
cutoff_hz = 0.5
taps=TOFandWebCam.DesignFilter(sample_rate ,tr_band,ripple_db,cutoff_hz)

## Apply filter
filtered_x_tof=QSR_DoubleLayer_TOFADCandPer.ApplyFilter(taps,x_tof_raw['1.0'])

FigureNum=plt.figure(num=2, figsize=(2.5, 1.5))
ax=[FigureNum.add_subplot(4,1,i) for i in range(1,5)]

initial=0
final= 14000 
# ax[0].plot(t_data['1.0'][initial:final],u_data['1.0'][initial:final,0],
#           alpha=0.7,
#           label='Input')
# ax[1].plot(t_data['1.0'][initial:final],x_adc['1.0'][initial:final,0],
#           alpha=0.7,
#           label='ADC')
# ax[2].plot(t_data['1.0'][initial:final],filtered_x_tof[initial:final,0],
#           alpha=0.7,
#           label='Filtered TOF')
# ax[3].plot(t_data['1.0'][initial:final],x_tof_online_fil['1.0'][initial:final,0],
#           alpha=0.7,
#           label='Online Filtered TOF') 

# =============================================================================
# Initialize SINDYc object
# =============================================================================
dt=1
tspan=t_data['1.0'][initial:final]

QSR_DoubleLayer_Obj_SINDYc_SI=SINDYc_MPC_Design('QSR_DoubleLayer_SINDYc_SI',tspan)
# =============================================================================
# Split training and test data
# =============================================================================

QSR_DoubleLayer_Obj_SINDYc_SI.SplitData2TrainAndValid(u=u_data['1.0'],
                                                      x=x_tof_online_fil['1.0'],
                                                      split=0.7,selection='defined')

u_QSR_DoubleLayer_train=QSR_DoubleLayer_Obj_SINDYc_SI.DataDictionary['u_train'] 
# u_valid=  
x_QSR_DoubleLayer_train=QSR_DoubleLayer_Obj_SINDYc_SI.DataDictionary['x_train']                                                   
tspan_train=tspan[0:x_QSR_DoubleLayer_train.shape[0]]
# tspan_valid=
# =============================================================================
#  # Compute noisy derivative by applying total variation regularisation
# =============================================================================
x_QSR_DoubleLayer_aug_TVRegDiffList=[]
dx_QSR_DoubleLayer_TVRegDiffList=[]
tspan_pred_List=[]
x_QSR_DoubleLayer_pred_List=[]

DiffrentiationType='TVRegDiff'
    
dx_QSR_DoubleLayer_TVRegDiff,x_QSR_DoubleLayer_aug_TVRegDiff=QSR_DoubleLayer_Obj_SINDYc_SI.ComputeDerivative(DiffrentiationType,
                                                     x_QSR_DoubleLayer_train,
                                                     u_QSR_DoubleLayer_train,
                                                     dt)
dx_QSR_DoubleLayer_TVRegDiffList.append(dx_QSR_DoubleLayer_TVRegDiff)
x_QSR_DoubleLayer_aug_TVRegDiffList.append(x_QSR_DoubleLayer_aug_TVRegDiff)

# =============================================================================
#  Apply Sparse regression on noisy data
# =============================================================================
#Define parameters
ModelName='SINDYc'
degree=4
usesine=0
#Sparse Regression

_,_,Xi_QSR_DoubleLayer=QSR_DoubleLayer_Obj_SINDYc_SI.TrainSINDYc(degree=4,
                                         x_aug=x_QSR_DoubleLayer_aug_TVRegDiff,
                                         dx=dx_QSR_DoubleLayer_TVRegDiff,
                                         lam_bda=0.0001,
                                         Normalization=True)

# =============================================================================
# Prediction over noisy data
# =============================================================================
DEParameters_SINDYc=(Xi_QSR_DoubleLayer,degree)
x0_QSR_DoubleLayer_pred=x_QSR_DoubleLayer_aug_TVRegDiff[0,0:1]


tspan_pred,x_QSR_DoubleLayer_pred=QSR_DoubleLayer_Obj_SINDYc_SI.ODEINTSimulation(QSR_DoubleLayer_Obj_SINDYc_SI.SparseGalerkinControl,
                                               x0_QSR_DoubleLayer_pred,
                                               x_QSR_DoubleLayer_aug_TVRegDiff[:,1],
                                               tspan,
                                               DEParameters_SINDYc)
tspan_pred_List.append(tspan_pred)
x_QSR_DoubleLayer_pred_List.append(x_QSR_DoubleLayer_pred)

# =============================================================================
# Plotting
# =============================================================================
for ii in range(0,1):
    # =============================================================================
    # noisy derivative
    # =============================================================================
    
    f2 = plt.figure(figsize=(2.5, 1.5))
    ax1 = f2.add_subplot(111)
    
    
    a12=ax1.plot(tspan[0:dx_QSR_DoubleLayer_TVRegDiff.shape[0]],
                 dx_QSR_DoubleLayer_TVRegDiffList[ii][:,0],color='darkred',ls='--',
                            alpha=1,
                            linewidth=1.5)
    
    
    ax1.set_xlabel(r"time t", size=12)
    ax1.set_ylabel(r"${\dot{x_1}}(t)$", size=12)
for ii in range(0,1):
    
    # =============================================================================
    # SINDYc Prediction on noisy and clean dataset
    # =============================================================================
    f1 = plt.figure(figsize=(2.5, 1.5))
    ax1 = f1.add_subplot(211)
    
    
    ax1.plot(tspan_train,u_QSR_DoubleLayer_train,color='limegreen',
                            alpha=0.8,
                            linewidth=1.5)
    
    ax1.set_xlabel(r"time", size=12)
    ax1.set_ylabel(r"${{u}}(t)$", size=12)
    
    ax2 = f1.add_subplot(212)
    ax2.plot(tspan_train,x_QSR_DoubleLayer_train,color='k',
                            alpha=0.8,
                            linewidth=1.5)

    
    ##Training data validation
    ax2.plot(tspan_pred_List[ii][0:x_QSR_DoubleLayer_pred_List[ii].shape[0]],
             x_QSR_DoubleLayer_pred_List[ii][:,0],color='skyblue',
                            alpha=0.8,
                            linewidth=1.5,
                            ls='--')
    
    # ax3 = f1.add_subplot(313)
    
    # ax3.plot(tspan_simu[0:9000],x_Lot_Vol_simu[500:-500,1],color='k',
    #                         alpha=0.8,
    #                         linewidth=1.5)
    
    # ax3.plot(tspan_pred_clean[0:9000],x_pred_clean[500:-495,1],color='darkorange',
    #                         alpha=0.8,
    #                         linewidth=1.5,
    #                         ls='--')
    
    # ax3.plot(tspan_pred_noisy_List[ii][0:9000],x_pred_noisy_List[ii][:,1],color='skyblue',
    #                         alpha=0.8,
    #                         linewidth=1.5,
    #                         ls='--')
    
    # ax2.set_xlabel(r"time", size=12)
    # ax2.set_ylabel(r"${\bf{x}}(t)$", size=12)
    
#    f1.savefig('Plots/NoisyDerivative/SINDYcPredCleanVsNoisy'+str(ii)+'.png', bbox_inches='tight',dpi=300)