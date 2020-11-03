# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:54:34 2020

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
import pysindy as ps

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
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
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
path = GenPath+"/DataFiles/Data20_10_2020_RoSE_layer_ADC/"
name = "TOFADCandPer"
Bolustype = "Dry"
RoSEv2pt0_3layer_TOFADCandPer = TOFandWebCam(path, "TOFADCandPer_RoSE_40mmat20mmps", name)
RoSEv2pt0_3layer_TOFADCandPer.find_all()
# Original category in PC QSR_DoubleLayer_TOFADCandPer.FileReading([12,13,6,14,15])
"""
Mac-->[0,1,2]
Pi--> 0,1,2
"""
RoSEv2pt0_3layer_TOFADCandPer.FileReading([0,1,2])
#12,13
dict_keys=list(RoSEv2pt0_3layer_TOFADCandPer.ArrayDict.keys())

# int_data=120
# # t_data={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][:,0]}
# t_data={'1.0':np.linspace(0,QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,0].shape[0],\
#             QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,0].shape[0])}
# u_data={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,[20]]}
# x_adc={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,[8]]}
# x_tof_raw={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,[2]]}
# x_tof_online_fil={'1.0':QSR_DoubleLayer_TOFADCandPer.ArrayDict[dict_keys[0]][int_data:,[4]]}

#Design filter
sample_rate = 10.0
tr_band=2
ripple_db = 25
cutoff_hz = 0.1
taps=TOFandWebCam.DesignFilter(sample_rate ,tr_band,ripple_db,cutoff_hz)

# ## Apply filter
# filtered_x_adc=QSR_DoubleLayer_TOFADCandPer.ApplyFilter(taps,x_adc['1.0'])

x_tof_online_fil_list=[]
x_adc_list=[]
u_data_list=[]
x_adc_filtered_list=[]
t_data_list=[]

for c, value in enumerate(dict_keys):
    """
    Index    File          int_data   num_of_dt_pt
    0        0_080         1100       5000
    1        0_100         1100       5000
    2        0_130         1100       5000
    4        0_140         1100       5000
    6        50_10_130     0170       10000
    13       60_100_130    0170 
    """
    layer_index=5
    int_data=(270,270,270,170,170,170)#(1100,1100,1100,1100,170)
    num_of_dt_pts=(1200,1200,1500,8000,8000)#(5000,5000,5000,5000,12000)  
    
    train_window=(int_data[c],int_data[c]+num_of_dt_pts[c])

    x_tof_online_fil_list=x_tof_online_fil_list+RoSEv2pt0_3layer_TOFADCandPer.ArrayDict[dict_keys[c]]\
                                 [train_window[0]:train_window[1],
                                 [20-10+layer_index,20-10+layer_index+1,20-10+layer_index+2]].tolist()
    x_adc_list=x_adc_list+RoSEv2pt0_3layer_TOFADCandPer.ArrayDict[dict_keys[c]]\
                                 [train_window[0]:train_window[1],
                                 [20+layer_index,20+layer_index+1,20+layer_index+2]].tolist()
    u_data_list= u_data_list+RoSEv2pt0_3layer_TOFADCandPer.ArrayDict[dict_keys[c]]\
                                  [train_window[0]:train_window[1],
                                  [20+12+layer_index,20+12+layer_index+1,20+12+layer_index+2]].tolist()
        
    ## Apply filter
    tempList=RoSEv2pt0_3layer_TOFADCandPer.ApplyFilter(taps,RoSEv2pt0_3layer_TOFADCandPer.ArrayDict[dict_keys[c]]\
                                 [0:-1,
                                 [20+layer_index,20+layer_index+1,20+layer_index+2]]).tolist()
    x_adc_filtered_list=x_adc_filtered_list+tempList[train_window[0]:train_window[1]]
      
time_scale_factor=10
t_data_list=t_data_list+np.arange(0,(len(u_data_list))/time_scale_factor,1/time_scale_factor).tolist()

FigureNum=plt.figure(num=1, figsize=(2.5, 1.5))
ax=[FigureNum.add_subplot(4,1,i) for i in range(1,5)]

initial=0
final= 40000

ax[0].plot(t_data_list[initial:final],u_data_list[initial:final],
          alpha=0.7,
          label='Input')
ax[1].plot(t_data_list[initial:final],x_adc_list[initial:final],
          alpha=0.7,
          label='ADC')
ax[2].plot(t_data_list[initial:final],x_adc_filtered_list[initial:final],
          alpha=0.7,
          label='Filtered TOF')
ax[3].plot(t_data_list[initial:final],x_tof_online_fil_list[initial:final],
          alpha=0.7,
          label='Online Filtered TOF') 

# =============================================================================
# Initialize SINDYc object
# =============================================================================
dt=0.1
tspan=np.array(t_data_list)



RoSEv2pt0_3layer_Obj_SINDYc_SI=SINDYc_MPC_Design('RoSEv2pt0_3layer_Obj_SINDYc_SI',tspan)
# # =============================================================================
# # Split training and test data
# # =============================================================================

# QSR_DoubleLayer_Obj_SINDYc_SI.SplitData2TrainAndValid(u=u_adc_offline_filtered,
#                                                       x=x_adc_offline_filtered,
#                                                       #x=x_tof_online_fil['1.0'],
#                                                       split=0.7,selection='defined')
N_train=500
u_RoSEv2pt0_3layer_train=np.array(u_data_list[0:N_train])
# u_QSR_DoubleLayer_test=QSR_DoubleLayer_Obj_SINDYc_SI.DataDictionary['u_valid']
u_RoSEv2pt0_3layer_all=np.array(u_data_list)
# # u_valid=  

x_RoSEv2pt0_3layer_tof_train=np.array(x_tof_online_fil_list[0:N_train])
x_RoSEv2pt0_3layer_all=np.array(x_tof_online_fil_list)

# x_QSR_DoubleLayer_test=QSR_DoubleLayer_Obj_SINDYc_SI.DataDictionary['x_valid']    
# x_QSR_DoubleLayer_all=x_adc_offline_filtered 

x_RoSEv2pt0_3layer_adc_train=np.array(x_adc_filtered_list[0:N_train])
x_RoSEv2pt0_3layer_adc_all=np.array(x_adc_filtered_list)
                                             
tspan_train=tspan[0:N_train]
tspan_all=tspan

threshold_vec=np.array([0.008])#0.008
#threshold_vec=np.arange(0.0001,0.02,0.0002)
for i in range(threshold_vec.shape[0]):
    
    try: 
       # =============================================================================
       # # Instantiate and fit the SINDYc model
       # =============================================================================
       # smoothedFD = ps.SmoothedFiniteDifference()
       
       stlsq_optimizer = ps.STLSQ(threshold=threshold_vec[i], alpha=.5,normalize=True)
       
       fourier_library = ps.FourierLibrary(n_frequencies=2)
       poly_library = ps.PolynomialLibrary(include_interaction=True,
                                           interaction_only=False,
                                           degree=2)
       combined_library=poly_library#+fourier_library
       
       
       # =============================================================================
       # Discrete-time model        
       # =============================================================================
       model_discrete = ps.SINDy(optimizer=stlsq_optimizer,
           feature_library=combined_library,
           discrete_time=True
           #differentiation_method=smoothedFD
           )
       
       model_discrete.fit(x_RoSEv2pt0_3layer_adc_train, 
                          u=u_RoSEv2pt0_3layer_train)
       model_discrete.print()
       
       # Creating a dummy test dataset from scaling train dataset
       u_RoSEv2pt0_3layer_dummy_test=u_RoSEv2pt0_3layer_all
        # np.concatenate((u_RoSEv2pt0_3layer_all,
        #                                               0.8 *(u_RoSEv2pt0_3layer_train[0:400,:]-40)+40,
        #                                               0.6 *(u_RoSEv2pt0_3layer_train[0:400,:]-40)+40,
        #                                               0.4 *(u_RoSEv2pt0_3layer_train[0:400,:]-40)+40,
        #                                               0.2 *(u_RoSEv2pt0_3layer_train[0:400,:]-40)+40,
        #                                               0.1 *(u_RoSEv2pt0_3layer_train[0:400,:]-40)+40
        #                                                    ))
       
       N=u_RoSEv2pt0_3layer_all.shape[0]
       x0_RoSEv2pt0_3layer_adc_all=x_RoSEv2pt0_3layer_adc_all[0,np.newaxis]
       # u_RoSEv2pt0_3layer_all_list=[u_RoSEv2pt0_3layer_all[:,0],u_RoSEv2pt0_3layer_all[:,1],u_RoSEv2pt0_3layer_all[:,2]]
       # x_RoSEv2pt0_3layer_adc_all__discrete_sim = model_discrete.simulate(x0_RoSEv2pt0_3layer_adc_all, 
       #                                               N, u_RoSEv2pt0_3layer_all[0:1000,:])
       N_all=u_RoSEv2pt0_3layer_dummy_test.shape[0]
       x_RoSEv2pt0_3layer_adc_all_sim=np.zeros((N_all,3))
       x_RoSEv2pt0_3layer_adc_all_sim[0,:]=x0_RoSEv2pt0_3layer_adc_all
       
       #Replacing the last two columns with zero to check redudancy
       #u_RoSEv2pt0_3layer_dummy_test[u_RoSEv2pt0_3layer_train.shape[0]:,1:3]=np.zeros((u_RoSEv2pt0_3layer_dummy_test[:,1:3].shape[0]-u_RoSEv2pt0_3layer_train.shape[0],
                                                                                       #u_RoSEv2pt0_3layer_dummy_test[:,1:3].shape[1]))
       # =============================================================================
       # Simulate the model with the entire dataset ie train+valid+dummy test
       # =============================================================================
       
       for ii in range(1,N_all):
           x_RoSEv2pt0_3layer_adc_all_sim[ii] = model_discrete.predict(x0_RoSEv2pt0_3layer_adc_all, 
                                                      u_RoSEv2pt0_3layer_dummy_test[ii-1:ii])
           # print(x_QSR_DoubleLayer_adc_all_sim[ii])
           x0_RoSEv2pt0_3layer_adc_all=x_RoSEv2pt0_3layer_adc_all_sim[ii,np.newaxis]
       
       #Saving as a csv file for filtered adc
       x_u_conc_fil_adc=np.hstack(( u_RoSEv2pt0_3layer_all,
                                x_RoSEv2pt0_3layer_adc_all
                                ))
       x_u_conc_fil_adc_df=pd.DataFrame(data=x_u_conc_fil_adc,
                                columns=['u0','u1','u2','x_fil0','x_fil1','x_fil2'])
       x_u_conc_fil_adc_df.to_csv('ControllerReferenceFiles/Reference_traj_filtered_adc_2.csv')    
       
       #Saving as a csv filefor simulated
       x_u_conc=np.hstack(( u_RoSEv2pt0_3layer_dummy_test,
                                x_RoSEv2pt0_3layer_adc_all_sim
                                ))
       x_u_conc_df=pd.DataFrame(data=x_u_conc,
                                columns=['u0','u1','u2','x_sim0','x_sim1','x_sim2'])
       x_u_conc_df.to_csv('ControllerReferenceFiles/Reference_traj_2.csv')
       
       FinPt1=u_RoSEv2pt0_3layer_train.shape[0]
       FinPt2=N_all-FinPt1
       FinPt3=u_RoSEv2pt0_3layer_all.shape[0]
       
       FigureNum=plt.figure(num=5+i, figsize=(2.5, 1.5))
       ax3=[FigureNum.add_subplot(4,1,i) for i in range(1,5)]
       
       ax3[0].plot(np.arange(0,FinPt2),u_RoSEv2pt0_3layer_dummy_test[0:FinPt2],
                 alpha=0.7,
                 label='Input')
       
       for i in range (1,4):
           ax3[i].plot(x_RoSEv2pt0_3layer_adc_all[0:FinPt1,i-1],#/np.max(x_QSR_DoubleLayer_tof_all[0:FinPt]),
                      label='true simulation')
           ax3[i].plot(np.arange(0,FinPt1),x_RoSEv2pt0_3layer_adc_all_sim[0:FinPt1,i-1],#/np.max(x_QSR_DoubleLayer_tof_all_sim[0:FinPt]),
                          label='model simulation')
           ax3[i].plot(np.arange(FinPt1,FinPt2),x_RoSEv2pt0_3layer_adc_all_sim[FinPt1:FinPt2,i-1],#/np.max(x_QSR_DoubleLayer_tof_all_sim[0:FinPt]),
                          'k',label='model simulation')
           ax3[i].plot(np.arange(FinPt1,FinPt3),x_RoSEv2pt0_3layer_adc_all[FinPt1:FinPt3,i-1],#/np.max(x_QSR_DoubleLayer_tof_all_sim[0:FinPt]),
                          'r',label='desired simulation')
       plt.show()
    
    except:
        print('A value in x_new is above the interpolation range for threshold=',threshold_vec[i])
 
