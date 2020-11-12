# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:56:23 2020

@author: useradmin-dbha483
"""

from scipy import zeros, signal, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cls():
    print("\n" * 50)


# clear Console
cls()
"""
t_delay=D_chamber/speed
t_delay=15mm/speed

if Ts=0.16s-->n=round(t_delay/0.16)

Wavelength
spatial_wavefrontlength=lambda/2=lamb_time/2 x speed
lamb_time=lambda/speed where lamb_time=2t_peak
lamb_time=80mm/20=4s, where spatial_wavefrontlength=80mm/2
lamb_time=120mm/20=6s
lamb_time=160mm/20=8s
"""
Ts=0.35
sample_rate = 1/Ts

start_time = 0
lamb_time=6

lamb_dis=int(lamb_time/Ts)


end_time = lamb_time+4

d = np.arange(start_time/Ts, end_time/Ts-Ts, 1)
x=0
epsilon=0 # Fixed offset for the wave
alpha=10
# d=0

#Calculating delay for manipulating the wavespeed
delay_18pt75mmps=15/18.75
delay_31pt25mmps=15/31.25
delay_46pt87mmps=15/46.87

delay=delay_18pt75mmps
delay_discrete=int(delay/Ts)
delay=delay_discrete
#Number of zeros to be embedded before each cycle of wave
const_embed=30 # 30 zeros are pre embedded
H=np.zeros((int(end_time/Ts)+1,5))
H=np.concatenate((epsilon*np.ones((const_embed,5)),H))
H[const_embed:,0]=d
H[const_embed:,1]=d
# sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

for i in range(0,d.shape[0]):
    if d[i]<=0*delay:
        H[const_embed+i,2]=epsilon        
    elif d[i]>=0*delay+lamb_dis:
        H[const_embed+i,2]=epsilon
    else:
       H[const_embed+i,2]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((0*delay-d[i])/lamb_dis))) 
       
for i in range(0,d.shape[0]):
    if d[i]<=delay:
        H[const_embed+i,3]=epsilon       
    elif d[i]>=delay+lamb_dis:
        H[const_embed+i,3]=epsilon
    else:
       H[const_embed+i,3]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((delay-d[i])/lamb_dis))) 

for i in range(0,d.shape[0]):
    if d[i]<=2*delay:
        H[const_embed+i,4]=epsilon
    elif d[i]>=2*delay+lamb_dis:
        H[const_embed+i,4]=epsilon
    else:
       H[const_embed+i,4]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((2*delay-d[i])/lamb_dis)))

# Varying amplitude
#xref=np.concatenate((H,
#                     H,
#                     0.75*(H-epsilon)+epsilon,
#                     0.75*(H-epsilon)+epsilon,
#                     0.5*(H-epsilon)+epsilon,
#                     0.5*(H-epsilon)+epsilon))
xref=H
for ii in range(0,50):
    xref=np.concatenate((xref,H))

       
xref_df=pd.DataFrame(xref)
xref_df.to_csv('ControllerReferenceFiles/Reference_traj_generated_1.csv')
FigureNum1=plt.figure(num=1, figsize=(2.5, 1.5))

ax1=FigureNum1.add_subplot(1,1,1)

ax1.plot(xref[:,2],
          alpha=0.7,c='r',
          label='Input')

ax1.plot(xref[:,3],
          alpha=0.7,c='green',
          label='Input')

ax1.plot(xref[:,4],
          alpha=0.7,c='blue',
          label='Input')

ax1.set_ylim(-1,15)


