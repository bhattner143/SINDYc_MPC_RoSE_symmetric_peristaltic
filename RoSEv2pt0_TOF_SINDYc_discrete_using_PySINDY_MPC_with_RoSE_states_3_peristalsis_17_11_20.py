
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:54:34 2020

@author: useradmin-dbha483
"""

import os
import pdb
import sys
import csv


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
import time
# from intelhex import IntelHex
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.nonparametric.smoothers_lowess import lowess
from dateutil.parser import parse
from scipy.signal import find_peaks, peak_prominences, kaiserord, lfilter, firwin, freqz
from scipy import zeros, signal, random
from scipy.optimize import minimize
from collections import deque
from matplotlib import style
from tvregdiff_master.tvregdiff import TVRegDiff
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sindybase import *
from ClassTOFandWebCam import *

from itertools import count
from matplotlib.animation import FuncAnimation
from RoSE_IOEXpander_TOF_cal_ten_OOPs import*

plt.style.use('fivethirtyeight')



#from Class_SINDYc_MPC_Design import *

def cls():
    print("\n" * 50)


# clear Console
cls()

# =============================================================================
# MPC design Class
# =============================================================================
class SINDYc_MPC_Design(SINDyBase,TOFandWebCam,RoSE_actuation_protocol):
    
    def __init__(self, path,filename,name,
                 SystemName_MPC,
                 UseIOExpander=False,
                 UseADC=False):
        
        TOFandWebCam.__init__(self,path,filename,name)
        RoSE_actuation_protocol.__init__(self,UseIOExpander=UseIOExpander,
                                         UseADC=UseADC)
#        self.path = path
        self.SystemName = SystemName_MPC
        self.DataDictionary={}
    
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
    
  
    def SplitData2TrainAndValid(self,u,x,tspan,split=0.8,selection='defined'):
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
            
        if tspan.ndim == 1:
            tspan= tspan[:, np.newaxis]
            
        
        if selection=='defined':
            
            _n = x.shape[0]
            train = np.arange(int(_n*split))
            valid = np.arange(int(_n*split),_n)
            x_train = x[train, :]
            x_valid =  x[valid, :]
            u_train = u[train, :]
            u_valid = u[valid, :]
            tspan_train=tspan[train, :]
            tspan_valid=tspan[valid, :]
            
            
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
                             'u_valid':u_valid,
                             'tspan_train':tspan_train,
                             'tspan_valid':tspan_valid
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
    
    def GenerateRobotTOFandADCData(self,
                                   int_data,
                                   num_of_dt_pts,
                                   time_scale_factor=1,
                                   *filter_params):        
        
        x_tof_online_fil_list=[]
        x_adc_list=[]
        u_data_list=[]
        x_adc_filtered_list=[]
        t_data_list=[]
        layer_index=5
        
        self.rob_data_dictionary={
                     't_data_list':t_data_list,
                     'u_data_list':u_data_list,
                     'x_adc_list':x_adc_list,
                     'x_adc_filtered_list':x_adc_filtered_list,
                     'x_tof_online_fil_list':x_tof_online_fil_list                     
                     }
        
        dict_keys=list(self.ArrayDict.keys())
        
        sample_rate,tr_band,ripple_db,cutoff_hz=filter_params[0]
        taps=self.DesignFilter(sample_rate ,tr_band,ripple_db,cutoff_hz)
        
        for c, value in enumerate(dict_keys):

            
            train_window=(int_data[c],int_data[c]+num_of_dt_pts[c])
            
            x_tof_online_fil_list=x_tof_online_fil_list+self.ArrayDict[dict_keys[c]]\
                                         [train_window[0]:train_window[1],
                                         [20-10+layer_index,20-10+layer_index+1,20-10+layer_index+2]].tolist()
            x_adc_list=x_adc_list+self.ArrayDict[dict_keys[c]]\
                                         [train_window[0]:train_window[1],
                                         [20+layer_index,20+layer_index+1,20+layer_index+2]].tolist()
            u_data_list= u_data_list+self.ArrayDict[dict_keys[c]]\
                                          [train_window[0]:train_window[1],
                                          [20+12+layer_index,20+12+layer_index+1,20+12+layer_index+2]].tolist()
                
            ## Apply filter
            tempList=self.ApplyFilter(taps,self.ArrayDict[dict_keys[c]]\
                                         [0:-1,
                                         [20+layer_index,20+layer_index+1,20+layer_index+2]]).tolist()
            x_adc_filtered_list=x_adc_filtered_list+tempList[train_window[0]:train_window[1]]
              
        time_scale_factor=10
        t_data_list=t_data_list+np.arange(0,(len(u_data_list))/time_scale_factor,1/time_scale_factor).tolist()
                
        self.rob_data_dictionary={
                     't_data_list':t_data_list,
                     'u_data_list':u_data_list,
                     'x_adc_list':x_adc_list,
                     'x_adc_filtered_list':x_adc_filtered_list,
                     'x_tof_online_fil_list':x_tof_online_fil_list                     
                     }
        
    def GenerateRobotSINDYcModel(self,
                         threshold_vec,
                         alpha=0.5,
                         degree=2,
                         trig_lib=False,
                         n_frequencies=2,
                         normalize=False,
                         smooth_diff=False,
                         #optimizer=stlsq_optimizer,
                         discrete_model=True,
                         model_print=True,
                         N_input=3,
                         N_states=3):
        # =============================================================================
        # # Instantiate and fit the SINDYc model
        # =============================================================================
        if smooth_diff is True:
            smoothedFD = ps.SmoothedFiniteDifference()
        
        stlsq_optimizer = ps.STLSQ(threshold=threshold_vec, 
                                   alpha=alpha,
                                   normalize=normalize)
        
        poly_library = ps.PolynomialLibrary(include_interaction=True,
                                            interaction_only=False,
                                            degree=degree)
        
        if trig_lib is True:            
            fourier_library = ps.FourierLibrary(n_frequencies=n_frequencies)
            library_sindyc=poly_library+fourier_library
        else:
            library_sindyc=poly_library
        
        if discrete_model is True:
            
            u_robot_train=self.DataDictionary['u_train'][:,0:N_input]
    
            x_robot_adc_train=self.DataDictionary['x_train'][:,0:N_states]
    
        
            self.model = ps.SINDy(optimizer=stlsq_optimizer,
                feature_library=library_sindyc,
                discrete_time=discrete_model
                #differentiation_method=smoothedFD
                )
            
            self.model.fit(x_robot_adc_train, u=u_robot_train)
            
        if model_print is True:
            self.model.print()
            
        else:
            print("CT model still need to be added")
            
        return self.model
            
    # =============================================================================
    #    Simulate SINDYc model         
    # =============================================================================
    def SimulateRobotSINDYcModel(self,x0,N,u):
        
        if self.model.get_params()['discrete_time'] is True:
            
            x_sindyc_model_sim = self.model.simulate(x0,N,u)
            
            self.DictSimulation={'x_sindyc_model_sim':x_sindyc_model_sim,
                                 'x0':x0,
                                 'N':N,
                                 'u':u}
        else:
            print("CT model sim still need to be added")
        
        return x_sindyc_model_sim
    # =============================================================================
    #  Predict SINDYc model   
    # =============================================================================
    def PredictRobotSINDYcModel(self,x0,u,N_hor=0):
    
        xk=x0
        
        # Make the dimension of xk atleast 1
        if xk.ndim==0:
            xk = xk[np.newaxis]
            
        if N_hor==0:
            N_hor=u.shape[0]    
            
        xk1_list=[]
        for ii in range(0,N_hor):
            xk1=self.model.predict(xk, u[ii])
            xk=xk1
            xk1_list.append(xk1[0])
        
        return np.array(xk1_list)
    # =============================================================================
    #  Generating data, SINDYc modeling and prediction function   
    # =============================================================================
    def DesignSINDYcModelForMPC(self,
                                file_index=[3],
                                filter_params=None,
                                N_train=500,
                                threshold_vec=np.array([0.008])):  
        
        self.find_all()
        
        # Original category in PC QSR_DoubleLayer_TOFADCandPer.FileReading([12,13,6,14,15])
        """Spyder 3 -->Raspberry Pi-->[16,14,17,2,10,0,4]
        Spyder 4 -->Other PC-->    [12,13,6,14,15,16,17]
        Spyder 4--> Mac-->[2,10,0,4]
        """
        self.FileReading(file_index)
        #12,13
        dict_keys=list(self.ArrayDict.keys())
        
        if filter_params==None:
            #Filter parameters
            sample_rate = 10.0
            tr_band=2
            ripple_db = 25
            cutoff_hz = 0.1
            filter_params=(sample_rate,tr_band,ripple_db,cutoff_hz)
                       
                 
        int_data=(270,270,270)
        num_of_dt_pts=(1200,1200,1200)
        
        #time span params
        time_scale_factor=10
            
        self.GenerateRobotTOFandADCData(int_data,
                                        num_of_dt_pts,
                                        time_scale_factor,
                                        filter_params,
                                        )       
        initial=0
        final= 40000
        
        t_data_list= self.rob_data_dictionary['t_data_list']
        u_data_list= self.rob_data_dictionary['u_data_list']
        x_adc_list= self.rob_data_dictionary['x_adc_list']
        x_adc_filtered_list= self.rob_data_dictionary['x_adc_filtered_list']
        x_tof_online_fil_list= self.rob_data_dictionary['x_tof_online_fil_list']
        
        # =============================================================================
        #  Applying TOF calibration obtained form WEbcam       
        # =============================================================================
        x_tof_online_fil_array=np.array(x_tof_online_fil_list)
        
        sf1=0.62+0.1
        sf2=0.62+0.05
        sf3=0.27+0.05

        x_tof_online_fil_array=np.array([sf1,sf2,sf3])*(x_tof_online_fil_array\
                                       -np.array([23,73,69]))+np.array([1,0,1])
        # =============================================================================
        # Split training and test data
        # =============================================================================
        train_data_length=N_train
        split_factor=train_data_length/len(u_data_list)
        
        self.SplitData2TrainAndValid(u=np.array(u_data_list),
                                                               x=x_tof_online_fil_array,
                                                               tspan=np.array(t_data_list),
                                                               #x=x_tof_online_fil['1.0'],
                                                               split=split_factor,
                                                               selection='defined')
        
        #Prepare training test and validation data
        u_robot_train=self.DataDictionary['u_train'][:,0:1]
        u_robot_test=self.DataDictionary['u_valid'][:,0:1]
        u_robot_valid=np.concatenate((u_robot_train,u_robot_test))
        
        x_robot_tof_train=self.DataDictionary['x_train'][:,0:1]
        x_robot_tof_test=self.DataDictionary['x_valid'][:,0:1]
        x_robot_tof_valid=np.concatenate((x_robot_tof_train,x_robot_tof_test))
        
        tspan_robot_train=self.DataDictionary['tspan_train'][:,0:1]
        tspan_robot_test =self.DataDictionary['tspan_valid'][:,0:1]
        tspan_robot_valid=np.concatenate((tspan_robot_train,tspan_robot_test))
        dt=0.1
        
    # =============================================================================
    #     Loop for checking threshold of different values
    # =============================================================================
        threshold_vec=threshold_vec#0.008
        #threshold_vec=np.arange(0.0001,0.001,0.00005)
        self.model_discrete_list=[]
        for i in range(threshold_vec.shape[0]):
            
            try:
                # =============================================================================
                # # Instantiate and fit the SINDYc model
                # =============================================================================
                model_discrete= self.GenerateRobotSINDYcModel(
                                  threshold_vec[i],
                                  alpha=0.5,
                                  degree=2,
                                  discrete_model=True)
                
                self.model_discrete_list.append(model_discrete)         
            except ValueError as v:
                print('A value in x_new is above the interpolation range for threshold=',v,threshold_vec[i])
        
        return self.model_discrete_list
    # =============================================================================
    #     
    # =============================================================================
    def filtering(self,x,z_adc_packed):
                z1,z2,z3=z_adc_packed
                x_filtered = np.array([[x[0,0],x[0,1],x[0,2]]])
                for index in range(0,10):
                    x_filtered[0,0],z1  = signal.lfilter(b_adc, 1, [x_filtered[0,0]], zi=z1)
                    x_filtered[0,1],z2  = signal.lfilter(b_adc, 1, [x_filtered[0,1]], zi=z2)
                    x_filtered[0,2],z3  = signal.lfilter(b_adc, 1, [x_filtered[0,2]], zi=z3)
                z_adc_packed=(z1,z2,z3)
                return x_filtered, z_adc_packed
        # =============================================================================
    #     Cost function of nonlinear MPC for Lotka-Volterra system
    # =============================================================================
    def RobotObjectiveFCN(self,u,*args):
    
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
        x,Ts,N,xref_all,u0,Q,R,Ru=args
        xk=x

        #Reshape the 1d array to to array
        u=u.reshape((3,-1))
        uk=u[:,0]
        J=0
        xref_k=xref_all[:,0]
        # Make the dimension of xk atleast 1
        if xk.ndim==0:
            xk = xk[np.newaxis]
            
        # if uk.ndim==0:
        #     uk=
        
    #    Loop through each prediction step
        for kk in range(N):
    #        Obtain plant state at next prediction step
            if xk.ndim==1:
                    xk_model=xk[:,np.newaxis] .T
            else:
                    xk_model=xk
            
            uk_model=uk
            uk_model=uk_model[:,np.newaxis] .T
            assert xk_model.shape[0]==uk_model.shape[0]==1, 'size of xk_model[0] and uk_model[0] must be same'
            xk1_model = self.model.predict(xk_model, uk_model)
            xk1=xk1_model[0,:]
            #J+=np.matmul(np.matmul((xk1-xref).T,Q),(xk1-xref))
            J+=(xk1-xref_k).T.dot(Q).dot((xk1-xref_k))
            
                
            if kk==0:
                J+=np.dot(np.dot((uk-u0).T,R),(uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
                
#                (uk-u0).T.dot(R).dot((uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
            else:
                J+=np.dot(np.dot((uk-u[:,kk-1]).T,R),(uk-u[:,kk-1]))+np.dot(np.dot(uk.T,Ru),uk)
                
            # Update xk and uk for the next prediction step
            
            xk=xk1
            
            if kk<N-1:
                uk=u[:,kk+1]
                xref_k=xref_all[:,kk+1]
                
                
        return J  
    
# =============================================================================
# Constraint function of nonlinear MPC for Lotka-Volterra system
# =============================================================================
                
    def RobotConstraintFCN(self,u,x,Ts,N):
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
            
            if xk.ndim==0:
                xk = xk[np.newaxis]
            
            for kk in range(N):
                #obtain new cart position at next prediction step
               
                xk1 = self.model.predict(xk, u)
                
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
#Main function: MPC Loop
# =============================================================================
#def main():
    
try:
    
    # =============================================================================
    # Path to Data directory and initialize the instant of SINDYc_MPC_Design class
    # =============================================================================
    GenPath = os.getcwd()
    path = GenPath+"/DataFiles/Data04_11_2020_RoSE_states_3_TOF_modeling/"
    name = "TOFADCandPer"
    Bolustype = "Dry"
    RoSEv2pt0_Obj_SINDYc_SI = SINDYc_MPC_Design(path, 
                                                    "TOFADCandPer_RoSEv2pt0_40mmat20mmps", 
                                                    name,
                                                    'RoSEv2pt0_DoubleLayer_SINDYc_SI',
                                                    UseIOExpander=True,
                                                    UseADC=True)
    # =============================================================================
    # Generating data from files, and designing a SINDYc model from the data
    # =============================================================================
    RoSEv2pt0_Obj_SINDYc_SI.DesignSINDYcModelForMPC(threshold_vec=np.array([0.008]))

    #Performance evaluating parameters
    perf_eval=np.array([[],[],[],[],[]]).T
    
    # Choose prediction horizon over which the optimization is performed
    Nvec=np.array([8,8,8,8,8])
    
    for i in range(Nvec.shape[0]):
        
        Ts          = .5              # Sampling time
        N           = Nvec[i]          # Control / prediction horizon (number of iterations)
        Duration    = 290            # Run control for 100 time units
        #Duration    = 545.5#418             # Run control for 100 time units
        
        Q           = 1*np.array([5,10,5])            # State weights
        #Q=Q[np.newaxis,:]
        R           = .5*np.array([1,1,1]) #0.5;         # Control variation du weights
        Ru = .5*np.array([1,1,1])#0.01                 # Control weights
    #    B = np.array([0,1])                    # Control vector (which state is controlled)
    #    B=B[:,np.newaxis]
    #    #C = eye(Nvar)                  # Measurement matrix
    #    D = 0                          # Feedforward (none)
        
        Ton      = 10                  # Time when control starts
        
        #Trajectory shaping parameters
        offset=0
        scale_fact=(10-offset)/(10-0)
        
        # Reference state, which shall be achieved
        # Reference trajectory, which shall be achieved
        xref_df_all1=pd.read_csv('ControllerReferenceFiles/TOF_peristalsis/Reference_traj_generated_1_18pt75mmps_40mm_Ts_0.16.csv')
#        xref_df_all1=pd.read_csv('ControllerReferenceFiles/TOF_peristalsis/Reference_traj_generated_1_18pt75mmps_40mm_cycles_50.csv')
        #xref_df_all1=pd.read_csv('ControllerReferenceFiles/TOF_peristalsis/Reference_traj_generated_2_46pt87mmps_40mm_cycles_50.csv')
        #xref_df_all1=pd.read_csv('ControllerReferenceFiles/TOF_peristalsis/Reference_traj_generated_8_lam_6_delay_2_Ts_0.35_cycles_50.csv')
#        xref_df_all3=pd.read_csv('ControllerReferenceFiles/TOF_peristalsis/Reference_traj_generated_6_9pt37mmps_80mm.csv')
        xref_all1=np.array(xref_df_all1)[:,3:]
#        xref_all2=np.array(xref_df_all2)[:,3:]
#        xref_all3=np.array(xref_df_all3)[0:,3:]
        
#        xref_all3=scale_fact*(xref_all3-66)+offset
        
        xref0 = 0*np.ones((3,100))#int(Ton/Ts))) #Initial part of the reference where the control is off
#        xref1 = xref_all[300:600,3:].T#2412:2542
#        xref2 = xref_all[1200:1500,3:].T
#        xref3 = xref_all[2550:2850,3:].T
        xref_traj=np.concatenate((xref0.T,xref_all1[0:-N,:]
#                                  xref_all2,
#                                  xref_all3[0:629,:]
                                  )).T
        
        assert xref_traj.shape[1]+N == int(Duration/Ts), 'size of xref_all must be equal to int(Duration/Ts)'
        
        N_states=xref_traj.shape[0]
        
        x0=xref_traj[:,0]#np.array([70])#x0_robot_adc_valid
        x0n=x0.T#[100; 50];             # Initial condition
        x        = x0n
        x=np.asarray(x).T
        
        uopt0 = np.array([9,9,9])                   # Set initial control input to thirty
        uopt     = uopt0[:,np.newaxis]*np.ones((N_states,N))
        uopt=np.asarray(uopt)
        
           
        xHistory = np.array([[],[],[]]).T       # Stores state history
        xfilteredHistory = np.array([[],[],[]]).T
        x_tofHistory=np.array([[],[],[]]).T 
        uHistory = np.array([[],[],[]]).T  # Stores control history
        tHistory = np.array([])       # Stores time history
        rHistory = np.array([[],[],[]]).T   # Stores reference (could be trajectory and vary with time)
        x_adc_filteredHistory=np.array([[],[],[]]).T
        uopt_all_layers_array=np.array([[],[],[],
                                        [],[],[],
                                        [],[],[],
                                        [],[],[]]).T
                                              
        d_stentHistory=np.array([[]]).T                                    

        #pdb.set_trace()
        
        # bound_optVar=[(LB,UB)]
        
        funval=np.zeros((int(Duration/Ts)))
        
        # RoSE Actuation parameters
        BaseLinePress=np.zeros((1,12))
        ScalingFact=1.0
        num_rose_layers=12
        uopt_all_deque=deque(np.zeros(57).tolist())
        
        #Filter Parameters
        numtaps_adc=10
        cutoff=0.1
        b_adc = signal.firwin(numtaps_adc, cutoff)
        z_adc = signal.lfilter_zi(b_adc, 1)
        z1=z2=z3=z_adc
        z_adc_packed=(z1,z2,z3)
        
        #Initialize TOF 
        t1 = timedelta(minutes = 0, seconds = 0, microseconds=0)
        
        Flag_UseFSP=False
        if RoSEv2pt0_Obj_SINDYc_SI.Flag_UseADC is True \
        and RoSEv2pt0_Obj_SINDYc_SI.Flag_UseIOExpander is True:
            
            TOFADCPer2dArray=np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).T
            range_mm_array=RoSEv2pt0_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)

            #Design real-time filter
            b,z=RoSE_actuation_protocol.Filter_RealTime_design(numtaps=10,cutoff=0.1)
            z=(z,z,z,z,z,z,z,z,z,z)
            TOF2dArray_mean_filtered_stacked=np.array([[],[],[],[],[],[],[],[],[],[],[]]).T
            
            if Flag_UseFSP is True:
                    RoSEv2pt0_Obj_SINDYc_SI.InitializeADCForFSP()
                    FSPADC2dArray=np.array([[],[],[]]).T
        
        start_time=time.time()
        t_in=timedelta(minutes = 0, seconds = 0, microseconds=0)
        
        
    
        # =============================================================================
        # #MPC loop
        # =============================================================================
        
        for ct in range(int(Duration/Ts)-2*N):
            
            if ct*Ts>30:   # Turn control on
                
                if ct*Ts==Ton+Ts:
                    print('Start Control.')
                
                # Set references
                #xref = np.asarray(xref1)
                xref = xref_traj[:,ct-1:ct-1+N]
                uopt0=uopt[:,0]
                uopt=uopt.flatten()
                
#                if ct*Ts==40:
#                    pdb.set_trace()
                ################################    
                if xref[0,0]>=7.6:
                    LB1=40 #Lower bound
                    UB1=46 #Upper bound
                    
#                elif xref[0,0]>=5.5 and xref[0,0]<7.6:
#                    LB1=32 #Lower bound
#                    UB1=40 #Upper bound
#                    
#                elif xref[0,0]>=2 and xref[0,0]<5.5:
#                    LB1=24 #Lower bound
#                    UB1=32 #Upper bound
                    
                else:
                    LB1=9
                    UB1=40#17
                #########################    
                if xref[1,0]>=8:
                    LB2=42 #Lower bound
                    UB2=46 #Upper bound
                    
#                elif xref[1,0]>=5.5 and xref[1,0]<8:
#                    LB2=34 #Lower bound
#                    UB2=40 #Upper bound
#                elif xref[1,0]>=2 and xref[1,0]<5.5:
#                    LB2=26 #Lower bound
#                    UB2=32  #Upper bound
                    

                else:
                    LB2=9
                    UB2=40#17
                 ###################################  
                if xref[2,0]>=7.5:
                    LB3=42 #Lower bound
                    UB3=46 #Upper bound                    
#                elif xref[2,0]>=6.5 and xref[2,0]<7.5:
#                    LB3=34 #Lower bound
#                    UB3=38 #Upper bound
#                elif xref[2,0]>=5 and xref[2,0]<6.5:
#                    LB3=32 #Lower bound
#                    UB3=34 #Upper bound
#                elif xref[2,0]>=4 and xref[2,0]<5:
#                    LB3=26 #Lower bound
#                    UB3=30 #Upper bound
                else:
                    LB3=9
                    UB3=42#17
            
                
                boundsopt=[(LB1,UB1),(LB2,UB2),(LB3,UB3)]

                boundsopt=[boundsopt[ii] for ii in range(0,N_states) for jj in range(0,N)]
                    
#                
                obj=RoSEv2pt0_Obj_SINDYc_SI
                
                uopt=minimize(obj.RobotObjectiveFCN,
                              uopt,
                              method='SLSQP',
                              args=(x,Ts,N,xref,uopt0,np.diag(Q),np.diag(R),np.diag(Ru)),
                              tol=0.1,
                              options={'ftol': 0.1, 'disp': False},
                              # constraints=cons,
                              bounds=boundsopt
                              )
                # pdb.set_trace()                
                
                funval[ct]=uopt.fun
                
                if np.absolute(x.any())>12:
                    break
                
                uopt=uopt.x
                
                uopt=uopt.reshape((3,-1))
                
            else:    
                
                if ct*Ts==0:
                    print('Control is off')
                    
                uopt     = uopt0[:,np.newaxis]*np.ones((N_states,N))
                    
                xref=-1000*np.ones((1,N_states))
                
            # Integrate system: Apply control & Step one timestep forward
            if x.ndim==1:
                x_model=x[:,np.newaxis].T
                
             #Apply input to RoSE
#            if ct*Ts==15:
#                pdb.set_trace()
#            uopt_all_layers=np.zeros((1,12))
#            uopt_all_layers[:,4:7]=uopt.T
            
            uopt_all_deque.rotate(3)
            uopt_all_deque.popleft()
            uopt_all_deque.popleft()
            uopt_all_deque.popleft()
            uopt_all_deque.popleft()
            uopt_all_deque.appendleft(uopt[2,0])
            uopt_all_deque.appendleft(uopt[1,0])
            uopt_all_deque.appendleft(uopt[0,0])
            uopt_all_deque.appendleft(0)
            
            u_deq_to_array=np.asarray(uopt_all_deque)
            uopt_all_layers=u_deq_to_array[[0,1,2,3,19,20,21,36,37,38,54,55]]+3
            
            RoSEv2pt0_Obj_SINDYc_SI.mergeDACadd2DataAndSend(uopt_all_layers[:,np.newaxis].T,
                                          0,
                                          BaseLinePress,
                                          ScalingFact,
                                          num_rose_layers,0)
            
            
            
#            assert uopt.shape==(N_states,N),'For SINDYC model to work, the shape of the uopt must be N_states x N'
#            uopt_model=uopt[:,0]
#            uopt_model=uopt_model[:,np.newaxis] .T
#            
#            assert x_model.shape[0]==uopt_model.shape[0], 'size of x_model[0] and uopt_model[0] must be same'
#            x_model_out = RoSEv2pt0_Obj_SINDYc_SI.model.predict(x_model,uopt_model)
#            x=x_model_out[0,:]
            
            #Generate ADC output from RoSE
            if RoSEv2pt0_Obj_SINDYc_SI.Flag_UseIOExpander is True \
            and RoSEv2pt0_Obj_SINDYc_SI.Flag_UseADC is True:
                
                
                pressure_kpa_array=RoSEv2pt0_Obj_SINDYc_SI.GeneratePressureDataFromADC(NoOfADC=12)
                
                range_mm_array=RoSEv2pt0_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)
                
                range_mm_array_mean=range_mm_array
                range_mm_array_filtered, z= RoSE_actuation_protocol.Filter_RealTime_apply(range_mm_array_mean[1:],
                                                                                      b,z)
                range_mm_array_filtered=range_mm_array_filtered[...,np.newaxis].T
                range_mm_array_filtered=np.concatenate((np.array([range_mm_array[0]]),
                                          range_mm_array_filtered[0]))
                
                if Flag_UseFSP is True:
                    try:
                        FSPADC2dArray=np.concatenate((FSPADC2dArray,np.array([[range_mm_array[0],
                                                                               RoSEv2pt0_Obj_SINDYc_SI.chanList[0].value,
                                                                               RoSEv2pt0_Obj_SINDYc_SI.chanList[1].value]])))
                    except ZeroDivisionError:
                        FSPADC2dArray=np.concatenate((FSPADC2dArray,np.array([[range_mm_array[0],
                                                                               0,
                                                                               0]])))
            
            #Generating Time stamp
            t_fi=np.array([RoSEv2pt0_Obj_SINDYc_SI.GenerateTimestamp(t1=t_in)])
            
            x_adc = np.array([pressure_kpa_array[4:7]])
            
            # Filter ADC data
            x_adc_filtered, z_adc_packed=RoSEv2pt0_Obj_SINDYc_SI.filtering(x_adc,z_adc_packed)
#            z=(z1,z2,z3)
#            x_filtered, z_adc_packed=filtering(x_adc,z_adc_packed)
            #Removing offsets
            x_adc_filtered[0,0]=x_adc_filtered[0,0]+15-3
            x_adc_filtered[0,1]=x_adc_filtered[0,1]+15.5-3
            x_adc_filtered[0,2]=x_adc_filtered[0,2]+12-.5
            
            
            x_tof=np.array([range_mm_array[[5,6,7]]])
            x_tof_filtered=np.array([range_mm_array_filtered[[5,6,7]]])
            
            d_stent=np.array([range_mm_array[[1]]])
            
            # =============================================================================
            # Applying TOF calibration obtained form WEbcam
            # =============================================================================
            sf1=1#=0.62#+0.1
            sf2=1#0.62#+0.05
            sf3=0.5#0.27+0.05
            
            x_tof_filtered=np.array([sf1,sf2,sf3])*(x_tof_filtered-np.array([23,73,69]))+np.array([1,0,1])
            #Additional offset tuning
            x_tof_filtered[:,0]=x_tof_filtered[:,0]+1.6+2.2+1
            x_tof_filtered[:,1]=x_tof_filtered[:,1]-3+2
            x_tof_filtered[:,2]=x_tof_filtered[:,2]-2.7
            
            x=x_tof_filtered
            
            xHistory=np.concatenate((xHistory,x))
            xfilteredHistory=np.concatenate((xfilteredHistory,x))
            x_tofHistory=np.concatenate((x_tofHistory,x_tof))#Unfiltered TOF data
            uHistory=np.vstack((uHistory,uopt[:,0]))
            tHistory = np.concatenate((tHistory,t_fi))
            x_adc_filteredHistory = np.concatenate((x_adc_filteredHistory,x_adc_filtered))#Filtered ADC for all layers
            #saving all control data for all layers
            uopt_all_layers_array=np.concatenate((uopt_all_layers_array,
                                                  uopt_all_layers[:,np.newaxis].T))
            #Stent displacement history
            d_stentHistory=np.concatenate((d_stentHistory,d_stent))
            
        exec_time=time.time() - start_time
        print("--- %s seconds ---" % (exec_time))
         
        
        # =============================================================================
        #   NRMSE      
        # =============================================================================
        error_rmse_tracking=np.sqrt(np.sum((xHistory[:,0:]-xref_traj[0:,0:xHistory[:,0].shape[0]].T)**2,axis=0)/\
        xHistory[:,0].shape[0])
        print("Tracking error-->",  error_rmse_tracking)
        
        perf_eval=np.concatenate((perf_eval,np.array([[N,exec_time,error_rmse_tracking[0],
                  error_rmse_tracking[1],
                  error_rmse_tracking[2]]])))
        # =============================================================================
         # Create a Dicionary for storing the data history
         # =============================================================================
        HistoryDict={
                  't':tHistory,
                  'x_ref_1':xref_traj[0,:],
                  'x_ref_2':xref_traj[1,:],
                  'x_ref_3':xref_traj[2,:],
                  'x_1':xHistory[:,0],
                  'x_2':xHistory[:,1],
                  'x_3':xHistory[:,2],#(5v/256*ADC_data-1)*125
                  'x_tof_1':x_tofHistory[:,0],
                  'x_tof_2':x_tofHistory[:,1],
                  'x_tof_3':x_tofHistory[:,2],
                  'x_adc_filtered_1':x_adc_filteredHistory[:,0],
                  'x_adc_filtered_2':x_adc_filteredHistory[:,1],
                  'x_adc_filtered_3':x_adc_filteredHistory[:,2],
                  'u_1':uHistory[:,0],  #1V-->50KPa and 1DAC-->0.0195V, 1DAC-->0.98KPa/step  
                  'u_2':uHistory[:,1], 
                  'u_3':uHistory[:,2], 
                  'd_stent':d_stentHistory[:,0], 
                  'J':funval
#                  'x_adc_all_raw_History_1':x_adc_all_raw_History[:,0],
#                  'x_adc_all_raw_History_2':x_adc_all_raw_History[:,1],
#                  'x_adc_all_raw_History_3':x_adc_all_raw_History[:,2],
#                  'x_adc_all_raw_History_4':x_adc_all_raw_History[:,3],
#                  'x_adc_all_raw_History_5':x_adc_all_raw_History[:,4],
#                  'x_adc_all_raw_History_6':x_adc_all_raw_History[:,5],
#                  'x_adc_all_raw_History_7':x_adc_all_raw_History[:,6],
#                  'x_adc_all_raw_History_8':x_adc_all_raw_History[:,7],
#                  'x_adc_all_raw_History_9':x_adc_all_raw_History[:,8],
#                  'x_adc_all_raw_History_10':x_adc_all_raw_History[:,9],
#                  'x_adc_all_raw_History_11':x_adc_all_raw_History[:,10],
#                  'x_adc_all_raw_History_12':x_adc_all_raw_History[:,11],
                        }
        
#        History_df = pd.DataFrame(HistoryDict) 
#        History_df.to_csv('DataFiles/Data06_11_2020_Peristalsis_experiment_with_MPC_with_TOF/data_controller_1.csv')
        
        #              'JHistory':funval}
        
#        uopt_all_layers_df=pd.DataFrame(uopt_all_layers_array)
#        uopt_all_layers_df.to_csv('DataFiles/Data06_11_2020_Peristalsis_experiment_with_MPC_with_TOF/u_data_controller_for_all_layers_1.csv')   
#        MPC_performance_evaluat_files 
        perf_eval_df=pd.DataFrame(perf_eval)
        perf_eval_df.to_csv('DataFiles/MPC_performance_evaluat_files/Performance_data_2')
except KeyboardInterrupt:
    print('Ctrl C Pressed')
  
finally:
    RoSE_clear=RoSE_actuation_protocol(UseIOExpander=False)
    ClearDAC=np.zeros((1,12),dtype=int)
    RoSE_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)
    print('All the best for next run')
    
    
 # =============================================================================
    #     Show results
    # =============================================================================
#    tspan=obj.DataDictionary['tspan_valid']
    f1 = plt.figure(num=1,figsize=(2.5, 1.5))
    ax1 = f1.add_subplot(311)

    ax1.plot(xHistory[:,0],
            color='red',
            linewidth=1.5,
            linestyle='-')
    ax1.plot(xref_traj[1,:],
                color='darkred',
                linewidth=1.5,
                linestyle='-',
                label='Output')
    
    ax2 = f1.add_subplot(312)

    ax2.plot(xHistory[:,1],
            color='green',
            linewidth=1.5,
            linestyle='-')
    ax2.plot(xref_traj[1,:],
                color='darkgreen',
                linewidth=1.5,
                linestyle='-',
                label='Output')
    
    ax3 = f1.add_subplot(313)

    ax3.plot(xHistory[:,2],
            color='blue',
            linewidth=1.5,
            linestyle='-')
    ax3.plot(xref_traj[1,:],
                color='darkblue',
                linewidth=1.5,
                linestyle='-',
                label='Output')
    
    f2 = plt.figure(num=2,figsize=(2.5, 1.5))
          
    ax4 = f2.add_subplot(311)
    ax4.plot(HistoryDict['u_1'],
                color='r',
                linewidth=1.5,
                linestyle='-',
                label='Control (u_1)')
    
    ax5 = f2.add_subplot(312)
    ax5.plot(HistoryDict['u_2'],
                color='g',
                linewidth=1.5,
                linestyle='-',
                label='Control (u_2)')
    
    ax6 = f2.add_subplot(313)
    ax6.plot(HistoryDict['u_3'],
                color='blue',
                linewidth=1.5,
                linestyle='-',
                label='Control (u_2)')
    

    f3 = plt.figure(num=3,figsize=(2.5, 1.5))
          
    ax7 = f3.add_subplot(311)
    ax7.plot(HistoryDict['x_adc_filtered_1'],
                color='r',
                linewidth=1.5,
                linestyle='-',
                label='')
    
    ax8 = f3.add_subplot(312)
    ax8.plot(HistoryDict['x_adc_filtered_2'],
                color='g',
                linewidth=1.5,
                linestyle='-',
                label='')
    
    ax9 = f3.add_subplot(313)
    ax9.plot(HistoryDict['x_adc_filtered_3'],
                color='blue',
                linewidth=1.5,
                linestyle='-',
                label='')
    
    ax1.set_ylim(-10, 15)
    ax2.set_ylim(-10, 15)
    ax3.set_ylim(-10, 15)
    
    ax4.set_ylim(0, 60)
    ax5.set_ylim(0, 60)
    ax5.set_ylim(0, 60)
    
    ax7.set_ylim(60, 75)
    ax8.set_ylim(60, 75)
    ax9.set_ylim(60, 75)
    
    ax1.set_xlabel(r"Time steps", size=12)
    ax1.set_ylabel(r"Amplitude (mm)", size=12)
    plt.show()
   
    #        # f1.savefig('Plots/MPC/MPC_N_'+str(i)+'.png', bbox_inches='tight',dpi=300)
#        f2 = plt.figure(num=8,figsize=(2.5, 1.5))
#        ax10 = f2.add_subplot(111)
#        ax10.plot(xfilteredHistory)
#        plt.show()
    

#if __name__=='__main__':
#    print('running within the module.\n')
#    main()
#else:
#    print('run from another module.\n')
#    main()
#    x_vals = []
#    y_vals = []
#    index = count()
#
#    ani = FuncAnimation(plt.gcf(), main, interval=1000)
#
#    plt.tight_layout()
#    plt.show()
