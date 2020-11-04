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
spatial_wavefrontlength=lamb_time/2 x speed
lamb_time=lambda/speed where lamb_time=2t_peak
lamb_time=40/10=4s
lamb_time=60/10=6s
lamb_time=80/10=8s
"""
Ts=0.16
sample_rate = 1/Ts

start_time = 0
lamb_time=8
end_time = lamb_time+4

d = np.arange(start_time, end_time, 1/sample_rate)
x=0
epsilon=66
alpha=8
# d=0


delay_18pt75mmps=15/18.75
delay_31pt25mmps=15/31.25
delay_46pt87mmps=15/46.87

delay=15/9.375
const_embed=30
H=np.zeros((int(end_time/Ts),5))
H=np.concatenate((66*np.ones((const_embed,5)),H))
H[const_embed:,0]=d
H[const_embed:,1]=d
# sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

for i in range(0,d.shape[0]):
    if d[i]<=0*delay:
        H[const_embed+i,2]=epsilon        
    elif d[i]>=0*delay+lamb_time:
        H[const_embed+i,2]=epsilon
    else:
       H[const_embed+i,2]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((0*delay-d[i])/lamb_time))) 
       
for i in range(0,d.shape[0]):
    if d[i]<=delay:
        H[const_embed+i,3]=epsilon       
    elif d[i]>=delay+lamb_time:
        H[const_embed+i,3]=epsilon
    else:
       H[const_embed+i,3]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((delay-d[i])/lamb_time))) 

for i in range(0,d.shape[0]):
    if d[i]<=2*delay:
        H[const_embed+i,4]=epsilon
    elif d[i]>=2*delay+lamb_time:
        H[const_embed+i,4]=epsilon
    else:
       H[const_embed+i,4]=epsilon+(alpha/2)*(1-np.cos(2*np.pi*((2*delay-d[i])/lamb_time)))


xref=np.concatenate((H,
                     H,
                     0.75*(H-66)+66,
                     0.75*(H-66)+66,
                     0.5*(H-66)+66,
                     0.5*(H-66)+66))

#xref=2.25*(xref-66)+56
xref_df=pd.DataFrame(xref)
#xref_df.to_csv('ControllerReferenceFiles/Reference_traj_generated_3.csv')
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

ax1.set_ylim(0,75)


