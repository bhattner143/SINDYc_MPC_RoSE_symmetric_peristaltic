#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:05:51 2019

@author: dipankarbhattacharya
"""

#!python
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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
       
      self.SINDYcTrainParam={'degree':degree,
                                 'lambda':lam_bda,
                                 'usesine':usesine}
      if x_aug==[] and dx==[]:
#          pdb.set_trace()
          
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

x0=np.array([100, 50])
xref = np.array([c/d,a/b]) # critical point

dt=0.01

tspan=np.linspace(dt, 100, 10000)

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

# ============================================================================= 
 # Compute clean derivative by applying central difference
# =============================================================================
DiffrentiationType='CentralDiff'

dx_CentralDiff,x_aug_CentralDiff=SINDYc_MPC_Design.ComputeDerivative(DiffrentiationType,
                                                     x_Lot_Vol_simu,
                                                     u,
                                                     dt,noise=False)
LotkaVolteraSysObj_SINDYc_SI.dx=dx_CentralDiff
LotkaVolteraSysObj_SINDYc_SI.x_aug=x_aug_CentralDiff
# =============================================================================
# Define SINDYc parameters and Apply Sparse regression on clean data
# =============================================================================
#Define parameters
ModelName='SINDYc'
degree=3
usesine=0
#Sparse Regression

_,_,Xi=LotkaVolteraSysObj_SINDYc_SI.TrainSINDYc(degree=3,
                                                x_aug=x_aug_CentralDiff,
                                                dx=dx_CentralDiff,
                                                lam_bda=0.001)
# =============================================================================
# Prediction over clean data
# =============================================================================
DEParameters_SINDYc_clean=(Xi,degree)
x0_pred_clean=x_aug_CentralDiff[0,0:2]
tspan_clean=tspan

tspan_pred_clean,x_pred_clean=LotkaVolteraSysObj_SINDYc_SI.ODEINTSimulation(LotkaVolteraSysObj_SINDYc_SI.SparseGalerkinControl,
                                               x0_pred_clean,
                                               x_aug_CentralDiff[:,2],
                                               tspan_clean,
                                               DEParameters_SINDYc_clean)


x2_Lot_Vol_simu_clean_std = np.std(x_Lot_Vol_simu_clean[:,1])
#eps_vec = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5])*x2_Lot_Vol_simu_clean_std
eps_vec = np.array([0.1])*x2_Lot_Vol_simu_clean_std
eps=eps_vec[0]


x_aug_NoisyList=[]
dx_TVRegDiffList=[]
x_aug_TVRegDiffList=[]
XiNoiseList=[]
tspan_pred_noisy_List=[]
x_pred_noisy_List=[]


# =============================================================================
# Generate noisy data from the clean data for comparison and plotting
# =============================================================================

for c,eps_val in enumerate(eps_vec):
    
    DiffrentiationType='CentralDiff'
    
    _,x_aug_Noisy=SINDYc_MPC_Design.ComputeDerivative(DiffrentiationType,
                                                         x_Lot_Vol_simu_clean,
                                                         u,
                                                         dt,noise=True,
                                                         eps=eps_val)
    x_aug_NoisyList.append(x_aug_Noisy)
    


for c,eps_val in enumerate(eps_vec):

    # =============================================================================
    # # Compute noisy derivative by applying total variation regularisation
    # =============================================================================
    
    
    
    
    DiffrentiationType='TVRegDiff'
    
    dx_TVRegDiff,x_aug_TVRegDiff=SINDYc_MPC_Design.ComputeDerivative(DiffrentiationType,
                                                         x_Lot_Vol_simu,
                                                         u,
                                                         dt,noise=True,
                                                         eps=eps_val)
    dx_TVRegDiffList.append(dx_TVRegDiff)
    x_aug_TVRegDiffList.append(x_aug_TVRegDiff)
    # =============================================================================
    #  Apply Sparse regression on noisy data
    # =============================================================================
    
    #Sparse Regression
    
    _,_,XiNoise=LotkaVolteraSysObj_SINDYc_SI.TrainSINDYc(degree=3,
                                             x_aug=x_aug_TVRegDiff,
                                             dx=dx_TVRegDiff,
                                             lam_bda=0.001)
    XiNoiseList.append(XiNoise)
    
    # =============================================================================
    # Prediction over noisy data
    # =============================================================================
    DEParameters_SINDYc_noisy=(XiNoise,degree)
    x0_pred_noisy=x_aug_TVRegDiff[0,0:2]
    tspan_noisy=tspan
    
    tspan_pred_noisy,x_pred_noisy=LotkaVolteraSysObj_SINDYc_SI.ODEINTSimulation(LotkaVolteraSysObj_SINDYc_SI.SparseGalerkinControl,
                                                   x0_pred_noisy,
                                                   x_aug_TVRegDiff[:,2],
                                                   tspan_noisy,
                                                   DEParameters_SINDYc_noisy)
    tspan_pred_noisy_List.append(tspan_pred_noisy)
    x_pred_noisy_List.append(x_pred_noisy)

print('---------SUCCESSFUL SIMULATION-------------')

# =============================================================================
# Plotting
# =============================================================================
for ii in range(eps_vec.shape[0]):
    # =============================================================================
    # ## Comparing the clean and noisy data
    # =============================================================================
    
    f3 = plt.figure(figsize=(2.5, 1.5))
    ax1 = f3.add_subplot(211)
    
    
    a12=ax1.plot(tspan[0:9995],
                 x_aug_NoisyList[ii][:,0],color='darkred',ls='--',
                            alpha=1,
                            linewidth=1.5)
    
    a11=ax1.plot(tspan[0:9995],
                 x_Lot_Vol_simu_clean[0:9995,0],color='k',
                            alpha=1,
                            linewidth=1.5)
    
    ax2 = f3.add_subplot(212)
    
    a22=ax2.plot(tspan[0:9995],
                 x_aug_NoisyList[ii][:,1],color='darkred',ls='--',
                            alpha=1,
                            linewidth=1.5)
    
    a21=ax2.plot(tspan[0:9995],
                 x_Lot_Vol_simu_clean[0:9995,1],color='k',
                            alpha=1,
                            linewidth=1.5)
    
    ax1.set_xlabel(r"time t", size=12)
    ax1.set_ylabel(r"${{x_1}}(t)$", size=12)
    
    ax2.set_xlabel(r"time t", size=12)
    ax2.set_ylabel(r"${{x_2}}(t)$", size=12)
    
    ax1.legend((a11[0], a12[0]), 
                       ('Clean','Noisy'),
    #                   bbox_to_anchor=(0., 1.02, 1., .202),
                       ncol=1,
                       prop={'size': 12},
    #                   mode="expand", 
                       borderaxespad=0,
                       handlelength=1,
                       labelspacing=0.1,
                       columnspacing=0.2,
                       loc=0,
                       frameon=False)
    
#    f3.savefig('Plots/NoisyDerivative/SINDYcCleanVsNoisyData'+str(ii)+'.png', bbox_inches='tight',dpi=300)

for ii in range(eps_vec.shape[0]):
    # =============================================================================
    # ## Comparing the clean and noisy derivative
    # =============================================================================
    
    f2 = plt.figure(figsize=(2.5, 1.5))
    ax1 = f2.add_subplot(211)
    
    
    a12=ax1.plot(tspan[0:dx_TVRegDiff.shape[0]],
                 dx_TVRegDiffList[ii][:,0],color='darkred',ls='--',
                            alpha=1,
                            linewidth=1.5)
    
    a11=ax1.plot(tspan[0:9000],
                 dx_CentralDiff[500:-495,0],color='k',
                            alpha=1,
                            linewidth=1.5)
    
    ax2 = f2.add_subplot(212)
    
    a22=ax2.plot(tspan[0:dx_TVRegDiff.shape[0]],
                 dx_TVRegDiffList[ii][:,1],color='darkred',ls='--',
                            alpha=1,
                            linewidth=1.5)
    
    a21=ax2.plot(tspan[0:9000],
                 dx_CentralDiff[500:-495,1],color='k',
                            alpha=1,
                            linewidth=1.5)
    
    ax1.set_xlabel(r"time t", size=12)
    ax1.set_ylabel(r"${\dot{x_1}}(t)$", size=12)
    
    ax2.set_xlabel(r"time t", size=12)
    ax2.set_ylabel(r"${\dot{x_2}}(t)$", size=12)
    
    ax1.legend((a11[0], a12[0]), 
                       ('Central difference','Total variation regularized derivative'),
    #                   bbox_to_anchor=(0., 1.02, 1., .202),
                       ncol=1,
                       prop={'size': 12},
    #                   mode="expand", 
                       borderaxespad=0,
                       handlelength=1,
                       labelspacing=0.1,
                       columnspacing=0.2,
                       loc=0,
                       frameon=False)
    
#    f2.savefig('Plots/NoisyDerivative/SINDYcCleanVsNoisy'+str(ii)+'.png', bbox_inches='tight',dpi=300)

for ii in range(eps_vec.shape[0]):
    
    # =============================================================================
    # SINDYc Prediction on noisy and clean dataset
    # =============================================================================
    f1 = plt.figure(figsize=(2.5, 1.5))
    ax1 = f1.add_subplot(311)
    
    
    ax1.plot(tspan_simu,u,color='limegreen',
                            alpha=0.8,
                            linewidth=1.5)
    
    ax1.set_xlabel(r"time", size=12)
    ax1.set_ylabel(r"${{u}}(t)$", size=12)
    
    ax2 = f1.add_subplot(312)
    ax2.plot(tspan_simu[0:9000],x_Lot_Vol_simu[500:-500,0],color='k',
                            alpha=0.8,
                            linewidth=1.5)
    
    
    ##Training data validation
    ax2.plot(tspan_pred_clean[0:9000],x_pred_clean[500:-495,0],color='darkorange',
                            alpha=0.8,
                            linewidth=1.5,
                            ls='--')
    
    ##Training data validation
    ax2.plot(tspan_pred_noisy_List[ii][0:9000],x_pred_noisy_List[ii][:,0],color='skyblue',
                            alpha=0.8,
                            linewidth=1.5,
                            ls='--')
    
    ax3 = f1.add_subplot(313)
    
    ax3.plot(tspan_simu[0:9000],x_Lot_Vol_simu[500:-500,1],color='k',
                            alpha=0.8,
                            linewidth=1.5)
    
    ax3.plot(tspan_pred_clean[0:9000],x_pred_clean[500:-495,1],color='darkorange',
                            alpha=0.8,
                            linewidth=1.5,
                            ls='--')
    
    ax3.plot(tspan_pred_noisy_List[ii][0:9000],x_pred_noisy_List[ii][:,1],color='skyblue',
                            alpha=0.8,
                            linewidth=1.5,
                            ls='--')
    
    ax2.set_xlabel(r"time", size=12)
    ax2.set_ylabel(r"${\bf{x}}(t)$", size=12)
    
#    f1.savefig('Plots/NoisyDerivative/SINDYcPredCleanVsNoisy'+str(ii)+'.png', bbox_inches='tight',dpi=300)