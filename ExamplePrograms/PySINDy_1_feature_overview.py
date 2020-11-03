# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:00:55 2020

@author: useradmin-dbha483
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Lasso

from scipy.interpolate import interp1d

import pysindy as ps


def cls():
    print("\n" * 50)


# clear Console
cls()

# Seed the random number generators for reproducibility
np.random.seed(100)
# Control input
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t ** 2])

# Lorenz equations with control input
def lorenz_control(z, t):
    u = u_fun(t)
    return [
        10 * (z[1] - z[0]) + u[0, 0] ** 2,
        z[0] * (28 - z[2]) - z[1],
        z[0] * z[1] - 8 / 3 * z[2] - u[0, 1],
    ]


# Generate measurement data
dt = .002

t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
x_train = odeint(lorenz_control, x0_train, t_train)
u_train = u_fun(t_train)

# Instantiate and fit the SINDy model
smoothedFD = ps.SmoothedFiniteDifference()


model = ps.SINDy(differentiation_method=smoothedFD)
model.fit(x_train, u=u_train, t=dt)
model.print()

# Evolve the Lorenz equations in time using a different initial condition
# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, 15, dt)
x0_test = np.array([8, 7, 15])
x_test = odeint(lorenz_control, x0_test, t_test)  
u_test = u_fun(t_test)

# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, u=u_test,t=dt))

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test,u=u_test)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(3.5, 4.5))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i],
                'k', label='numerical derivative')
    axs[i].plot(t_test, x_dot_test_predicted[:, i],
                'r--', label='model prediction')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
fig.show()

u_test = u_fun(t_test)
u_interp = interp1d(t_test, u_test, axis=0, kind='cubic')


# Evolve the new initial condition in time with the SINDy model.
# We trim off one time point because scipy.odeint tries to evaluate u_interp
# just past t_test[-1], where u_interp is not defined.
x_test_sim = model.simulate(x0_test, t_test[:-1], u=u_interp)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(3.5, 4.5))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test[:-1], x_test[:-1, i], 'k', label='true simulation')
    axs[i].plot(t_test[:-1], x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))

fig = plt.figure(figsize=(3.5, 4.5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
ax1.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='true simulation')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], 'r--')
ax2.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='model simulation')

fig.show()