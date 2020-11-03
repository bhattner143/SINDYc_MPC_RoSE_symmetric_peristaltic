
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
from datetime import timedelta,datetime
from scipy.signal import find_peaks, peak_prominences, kaiserord, lfilter, firwin, freqz
from scipy import zeros, signal, random
from scipy.optimize import minimize
from matplotlib import style
from tvregdiff_master.tvregdiff import TVRegDiff
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sindybase import *
from ClassTOFandWebCam import *
from RoSE_IOEXpander_TOF_cal_ten_OOPs import*

from itertools import count
from matplotlib.animation import FuncAnimation

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
#        self.tspan=tspan
#        self.n_tspan=self.tspan.shape[0]
#        self.x=np.zeros((self.n_tspan,2))
    
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
                                         [10]].tolist()
            x_adc_list=x_adc_list+self.ArrayDict[dict_keys[c]]\
                                         [train_window[0]:train_window[1],
                                         [23]].tolist()
            u_data_list= u_data_list+self.ArrayDict[dict_keys[c]]\
                                          [train_window[0]:train_window[1],
                                          [23+12]].tolist()
                
            ## Apply filter
            tempList=self.ApplyFilter(taps,self.ArrayDict[dict_keys[c]]\
                                         [0:-1,
                                         [23]]).tolist()
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
                         N_input=1,
                         N_states=1):
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
                                file_index=[0],
                                filter_params=None,
                                N_train=2200,
                                threshold_vec=np.array([0.0006])):  
        
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
            tr_band=1
            ripple_db = 10
            cutoff_hz = 0.1
            filter_params=(sample_rate,tr_band,ripple_db,cutoff_hz)
                       
        #Data initilization and length params  
        """
        Index    File          int_data   num_of_dt_pt
            0        0_080         1100       5000
            1        0_100         1100       5000
            2        0_130         1100       5000
            4        0_140         1100       5000
            6        50_10_130     0170       10000
            13       60_100_130    0170 
        """             
        int_data=(170,170,170)#(1100,1100,1100,1100,170)
        num_of_dt_pts=(8000,8000,8000)#(5000,5000,5000,5000,12000)  
        
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
        # Split training and test data
        # =============================================================================
        train_data_length=N_train
        split_factor=train_data_length/len(u_data_list)
        
        self.SplitData2TrainAndValid(u=np.array(u_data_list),
                                                               x=np.array(x_adc_filtered_list),
                                                               tspan=np.array(t_data_list),
                                                               #x=x_tof_online_fil['1.0'],
                                                               split=split_factor,
                                                               selection='defined')
        
        #Prepare training test and validation data
        u_robot_train=self.DataDictionary['u_train'][:,0:1]
        u_robot_test=self.DataDictionary['u_valid'][:,0:1]
        u_robot_valid=np.concatenate((u_robot_train,u_robot_test))
        
        x_robot_adc_train=self.DataDictionary['x_train'][:,0:1]
        x_robot_adc_test=self.DataDictionary['x_valid'][:,0:1]
        x_robot_adc_valid=np.concatenate((x_robot_adc_train,x_robot_adc_test))
        
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
        x,Ts,N,xref,u0,Q,R,Ru=args
        xk=x

        #Reshape the 1d array to to array
        u=u.reshape((1,-1))
        uk=u[:,0]
        J=0
        
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
                xk_model=xk.T
            
            uk_model=uk
            uk_model=uk_model[:,np.newaxis] .T
            
            assert xk_model.shape[0]==uk_model.shape[0]==1, 'size of xk_model[0] and uk_model[0] must be same'
            xk1_model = self.model.predict(xk_model, uk_model)
            xk1=xk1_model[0,:]
            #J+=np.matmul(np.matmul((xk1-xref).T,Q),(xk1-xref))
            J+=(xk1-xref).T.dot(Q).dot((xk1-xref))
            
                
            if kk==0:
                J+=np.dot(np.dot((uk-u0).T,R),(uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
                
#                (uk-u0).T.dot(R).dot((uk-u0))+np.dot(np.dot(uk.T,Ru),uk)
            else:
                J+=np.dot(np.dot((uk-u[:,kk-1]).T,R),(uk-u[:,kk-1]))+np.dot(np.dot(uk.T,Ru),uk)
                
            # Update xk and uk for the next prediction step
            
            xk=xk1
            
            if kk<N-1:
                uk=u[:,kk+1]
                
                
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
    path = GenPath+"/DataFiles/Data07_10_2020_RoSE_For_ADC_and_TOF_model/"
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
    RoSEv2pt0_Obj_SINDYc_SI.DesignSINDYcModelForMPC(threshold_vec=np.array([0.002]))

    # Choose prediction horizon over which the optimization is performed
    Nvec=np.array([2])
    
    for i in range(Nvec.shape[0]):
        
        Ts          = .1              # Sampling time
        N           = Nvec[i]          # Control / prediction horizon (number of iterations)
        Duration    = 180#530             # Run control for 100 time units
        
        Q           = np.array([1])            # State weights
        #Q=Q[np.newaxis,:]
        R           = 0.5*np.array([1]) #0.5;         # Control variation du weights
        Ru = 0.5*np.array([1])#0.01                 # Control weights
    #    B = np.array([0,1])                    # Control vector (which state is controlled)
    #    B=B[:,np.newaxis]
    #    #C = eye(Nvar)                  # Measurement matrix
    #    D = 0                          # Feedforward (none)
        
        Ton      = 30                  # Time when control starts
        
        # Reference state, which shall be achieved
        # Reference trajectory, which shall be achieved
        #xref_df_all=pd.read_csv('ControllerReferenceFiles/Reference_traj_adc_symmetric_const_Staircase.csv')
        xref_df_all=pd.read_csv('ControllerReferenceFiles/Reference_traj_adc_for_FSP_testing.csv')
        xref_all=np.array(xref_df_all)[:,0:]
        xref0 = 0*np.ones((1,300))#int(Ton/Ts))) #Initial part of the reference where the control is off
#        xref1 = xref_all[0:800,1:].T#5000,1:].T
        
        xref_val=np.array([63,70,63])
        xref_rep=np.repeat(xref_val,100)
        xref_rep_cycle=np.array([(np.ones((5,1))*xref_rep).flatten()]).T
        xref_traj=np.concatenate((xref0.T,xref_rep_cycle)).T
        
        assert xref_traj.shape[1] == int(Duration/Ts), 'size of xref_all must be equal to int(Duration/Ts)'
        
        N_states=xref_traj.shape[0]
        
        x0=xref_traj[:,0]#np.array([70])#x0_robot_adc_valid
        x0n=x0.T#[100; 50];             # Initial condition
        x        = x0n
        x=np.asarray(x).T
        
        uopt0 = np.array([10])                   # Set initial control input to thirty
        uopt     = uopt0[:,np.newaxis]*np.ones((N_states,N))
        uopt=np.asarray(uopt)
        
        # if uopt.shape[1]==1:
        #     uopt=uopt[:,0]
        
        # Constraints on control optimization
        
        # LB = 50#*np.ones((1,3))#[] #-100*ones(N,1);        # Lower bound of control input
        # UB = 100#*np.ones((1,3))           # Upper bound of control input
        
        xHistory = np.array([])       # Stores state history
        x_tofHistory=np.array([]) 
        x_tof_filteredHistory=np.array([]) 
        uHistory = np.array([]) # Stores control history
        tHistory = np.array([])       # Stores time history
        rHistory = np.array([])  # Stores reference (could be trajectory and vary with time)
        
        #pdb.set_trace()
        
        # bound_optVar=[(LB,UB)]
        
        funval=np.zeros((int(Duration/Ts)))
                    
        # RoSE Actuation parameters
        BaseLinePress=np.zeros((1,12))
        ScalingFact=1.0
        num_rose_layers=12
        
        #Initialize TOF 
        t1 = timedelta(minutes = 0, seconds = 0, microseconds=0)
        
        Flag_UseFSP=True
        if RoSEv2pt0_Obj_SINDYc_SI.Flag_UseADC is True \
        and RoSEv2pt0_Obj_SINDYc_SI.Flag_UseIOExpander is True:
            
            TOFADCPer2dArray=np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).T
            range_mm_array=RoSEv2pt0_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)

            #Design real-time filter
            b,z=RoSE_actuation_protocol.Filter_RealTime_design(numtaps=50 )
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
        for ct in range(int(Duration/Ts)):  
            
            if ct*Ts>30:   # Turn control on
                
                if ct*Ts==Ton+Ts:
                    print('Start Control.')
            
                # Set references
                #xref = np.asarray(xref1)
                xref = xref_traj[:,ct-1]
                uopt0=uopt[:,0]
                uopt=uopt.flatten()
                                
#                pdb.set_trace()
                if xref[0]>=62 and xref[0]<66:
                    LB1=2 #Lower bound
                    UB1=11
#                if xref[0]>=66 and xref[0]<67:
#                    LB1=12 #Lower bound
#                    UB1=14
                elif xref[0]>=64 and xref[0]<67:
                    LB1=9#Lower bound
                    UB1=15 #Upper bound
                elif xref[0]>=67 and xref[0]<69:
                    LB1=15 #Lower bound
                    UB1=21 #Upper bound
                elif xref[0]>=69 and xref[0]<71:
                    LB1=23
                    UB1=27
                elif xref[0]>=71 and xref[0]<=73:
                    LB1=29
                    UB1=33
                
                # else:
                #     LB=100
                #     UB=105
#                    else:#if xref[0]>=59.8 and xref[0]<=62:
#                        LB1=119
#                        UB1=121#92.5
                    
                if N==1:
                    boundsopt=[(LB1,UB1)]
                    
                    
                elif N>1:     
                    boundsopt=[(LB1,UB1),(LB1,UB1)]#,(LB1,UB1),(LB1,UB1)
                               #(LB1,UB1),(LB1,UB1),(LB1,UB1),(LB1,UB1)
                               #]
                    
                    
                    
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
                
                if np.absolute(x[0])>700:
                    break
                
                uopt=uopt.x
                
                uopt=uopt.reshape((1,-1))
                
            else:    #% If control is off
                
                # uopt=uopt0*np.ones((N))
                if ct*Ts==0:
                    print('Control is off')
                uopt     = uopt0[:,np.newaxis]*np.ones((N_states,N))
                    
                xref=-1000*np.ones((1,N_states))
                
            # Integrate system: Apply control & Step one timestep forward
            if x.ndim==1:
                x_model=x[:,np.newaxis].T
                            
            #Apply input to RoSE
            uopt0_all_layers=uopt[:,0]*np.ones((1,num_rose_layers))+2
            RoSEv2pt0_Obj_SINDYc_SI.mergeDACadd2DataAndSend(uopt0_all_layers,
                                          0,
                                          BaseLinePress,
                                          ScalingFact,
                                          num_rose_layers,0)
            
            
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
            
            x = np.array([[pressure_kpa_array[1]]])
            x_tof=np.array([[range_mm_array[2]]])
            x_tof_filtered=np.array([[range_mm_array_filtered[2]]])
            
#            t_History=tHistory[-1]+Ts/1
            xHistory=np.concatenate((xHistory,x[:,0]))
            x_tofHistory=np.concatenate((x_tofHistory,x_tof[:,0]))
            x_tof_filteredHistory=np.concatenate((x_tof_filteredHistory,x_tof_filtered[:,0]))
            uHistory=np.concatenate((uHistory,uopt[:,0]))
            tHistory = np.concatenate((tHistory,t_fi))
            
            if xref.ndim==2:
                rHistory = np.concatenate((rHistory,xref[:,0]))
            else:
                rHistory = np.concatenate((rHistory,xref))
            


        print("--- %s seconds ---" % (time.time() - start_time))
         
        
         # =============================================================================
         # Create a Dicionary for storing the data history
         # =============================================================================
        HistoryDict={
                  't':tHistory,
                  'x_ref_1':(5/256*rHistory-1)*125,
                  'x_1':(5/256*xHistory-1)*125,#(5v/256*ADC_data-1)*125
                  'x_tof':x_tofHistory,
                  'x_tof_filtered':x_tof_filteredHistory,
                  'u_1':uHistory,  #1V-->50KPa and 1DAC-->0.0195V, 1DAC-->0.98KPa/step                                        
                  'J':funval,
                  'FSP_reading': FSPADC2dArray[:,2]/(FSPADC2dArray[:,1]-FSPADC2dArray[:,2])
                  }
        
        History_df = pd.DataFrame(HistoryDict) 
        History_df.to_csv('DataFiles/Data15_10_2020_stent_FSP_experiment_with_MPC/data_controller_1.csv')
        
        
        # =============================================================================
        #     Show results
        # =============================================================================
        tspan=obj.DataDictionary['tspan_valid']
        f1 = plt.figure(num=7,figsize=(2.5, 1.5))
        ax1 = f1.add_subplot(311)
         
        ax1.plot(np.array([Ton+tspan[0],Ton+tspan[0]]),np.array([15,260]),
                  color='limegreen',
                    alpha=0.8,
                    linestyle='--',
                    linewidth=1.5)
        
#        ax1.plot(tHistory+tspan[0],np.zeros((tHistory.shape[0])),
#                color='k',
#                linewidth=1.5,
#                linestyle='--')
        
        ax1.plot(HistoryDict['x_ref_1'],
                color='skyblue',
                linewidth=1.5,
                linestyle='--')
        
        ph0 = ax1.plot(HistoryDict['x_1'],
                    color='darkblue',
                    linewidth=1.5,
                    linestyle='-',
                    label='Output')
        
        ax2 = f1.add_subplot(312)
        ax2.plot(HistoryDict['u_1'],
                    color='r',
                    linewidth=1.5,
                    linestyle='-',
                    label='Control (u_1)')
        
        ax3 = f1.add_subplot(313)
        ax3.plot(HistoryDict['x_tof_filtered'],
                    color='r',
                    linewidth=1.5,
                    linestyle='-',
                    label='TOF_filetered')
        # ax1.legend(
        #                     ncol=1,
        #                     prop={'size': 12},
        # #                   mode="expand", 
        #                     borderaxespad=0,
        #                     handlelength=1,
        #                     labelspacing=0.1,
        #                     columnspacing=0.2,
        #                     loc=0,
        #                     frameon=False)
        
        # handles, labels = ax1.get_legend_handles_labels()
        ax1.set_ylim(0, 100)
#        ax1.set_xlim(0, 1100)
#        ax2.set_xlim(0, 1100)
        ax3.set_ylim(90, 110)
#        ax3.set_xlim(0, 1100)
#        ax2.set_ylim(0, 50)
        ax1.set_xlabel(r"Time steps", size=12)
        ax1.set_ylabel(r"Pressure (Kpa)", size=12)
        
        ax2.set_xlabel(r"Time steps", size=12)
        ax2.set_ylabel(r"Digital values for DAC", size=12)
        
        plt.show()
        
        # f1.savefig('Plots/MPC/MPC_N_'+str(i)+'.png', bbox_inches='tight',dpi=300)
        
        
        
except KeyboardInterrupt:
    print('Ctrl C Pressed')
    
finally:
    RoSE_clear=RoSE_actuation_protocol(UseIOExpander=False)
    ClearDAC=np.zeros((1,12),dtype=int)
    RoSE_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)
    print('All the best for next run')
        

#if __name__=='__main__':
#    print('running within the module.\n')
#    main()
#else:
#    print('run from another module.\n')
