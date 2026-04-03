#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:37:51 2019

@author: dipankarbhattacharya
"""

import numpy as np
from parametric_pde_find import *
from scipy.integrate import odeint
from numpy.fft import fft, ifft, fftfreq
from time import time
import matplotlib.pyplot as plt

fontsize = 20


def cls():
    print("\n" * 50)

def parametric_burgers_rhs(u, t, params):
    k,a,b,c = params
    deriv = a*(1+c*np.sin(t))*u*ifft(1j*k*fft(u)) + b*ifft(-k**2*fft(u))
    return np.real(deriv)


# Set size of grid
n = 256
m = 256

# Set up grid
x = np.linspace(-8,8,n+1)[:-1];   dx = x[1]-x[0]
t = np.linspace(0,10,m);          dt = t[1]-t[0]
k = 2*np.pi*fftfreq(n, d = dx)

# Initial condition
u0 = np.exp(-(x+1)**2)

# Solve with time dependent uu_x term
params = (k, -1, 0.1, 0.25)
u = odeint(parametric_burgers_rhs, u0, t, args=(params,)).T

u_xx_true = 0.1*np.ones(m)
uu_x_true = -1*(1+0.25*np.sin(t))

# Plot
FigureNum1 = plt.figure(num=1, figsize=(16, 4))
X, T = np.meshgrid(x, t)

ax = FigureNum1.subplots(1, 2)


ax[0].pcolor(X, T, u.T, cmap='coolwarm')
ax[0].set_xlabel('x', fontsize = fontsize)
ax[0].set_ylabel('t', fontsize = fontsize)
#plt.xticks(fontsize = fontsize)
#plt.yticks(fontsize = fontsize)
#ax[0].xlim([x[0],x[-1]])

ax[1].plot(t, uu_x_true, label=r'$uu_{x}$')
ax[1].plot(t, u_xx_true, label=r'$u_{xx}$')

#xticks(fontsize = fontsize)
#yticks(fontsize = fontsize)
ax[1].set_xlabel('t', fontsize = fontsize)
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = fontsize+2)

Ut, Theta, rhs_des = build_linear_system(u, dt, dx, D=4, P=3, time_diff = 'FD', space_diff = 'FD')