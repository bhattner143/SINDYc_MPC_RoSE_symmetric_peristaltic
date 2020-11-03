#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:23:16 2019

@author: dipankarbhattacharya
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

from matplotlib import style
import pandas as pd
# from intelhex import IntelHex
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.nonparametric.smoothers_lowess import lowess
from dateutil.parser import parse
from scipy.signal import find_peaks, peak_prominences, kaiserord, lfilter, firwin, freqz
from scipy import zeros, signal, random
import random
from ClassTOFandWebCam import *



#from Class_SINDYc_MPC_Design import *


def cls():
    print("\n" * 50)


# clear Console
cls()

# Program begins for check

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

##
# =============================================================================
# Time of FLight
# =============================================================================
GenPath = os.getcwd()
path = GenPath+"/DataFiles/Data18_03_2020/TOF"
name = "TOF"
Bolustype = "Dry"
QSR_DoubleLayer_TOF = TOFandWebCam(path, "stent_P1537_P0_40mm", name)
QSR_DoubleLayer_TOF.find_all()
QSR_DoubleLayer_TOF.FileReading([1,0,2])

dict_keys=list(QSR_DoubleLayer_TOF.ArrayDict.keys())

x_tof={'0.6':QSR_DoubleLayer_TOF.ArrayDict[dict_keys[0]],
            '0.8':QSR_DoubleLayer_TOF.ArrayDict[dict_keys[1]],
            '1.0':QSR_DoubleLayer_TOF.ArrayDict[dict_keys[2]]}


TOFandWebCam.DispPlotting(plt.figure(num=1, figsize=(5, 3)),
                          3,
                          x_tof['0.6'],x_tof['0.8'],x_tof['1.0'],
                          window=[(300,900),(300,900),(300,900)],
                          label=['a','b','c']
                          )

#Design filter
taps=TOFandWebCam.DesignFilter()

## Apply filter
QSR_DoubleLayer_TOF.ApplyFilter(taps)

filtered_x_tof=QSR_DoubleLayer_TOF.ArrayDictFiltered#['QSR_2Layer_TOF_test_0pt6_40mm_20mmps']

# #############################################################################
# Fit IsotonicRegression and LinearRegression models

ir_x_tof={'0.6':TOFandWebCam.Detrending_isotonicRegression(filtered_x_tof[dict_keys[0]]),
          '0.8':TOFandWebCam.Detrending_isotonicRegression(filtered_x_tof[dict_keys[1]]),
          '1.0':TOFandWebCam.Detrending_isotonicRegression(filtered_x_tof[dict_keys[2]])}

TOFandWebCam.DispPlotting(plt.figure(num=2, figsize=(5, 3)),
                          3,
                          ir_x_tof['0.6'],ir_x_tof['0.8'],ir_x_tof['1.0'],
                          window=[(300,900),(300,900),(300,900)],
                          label=['a','b','c']
                          )



# # ##Plot
# # InPoint=100
# # NoDatapoints=1300                             
# # FigureNum2 = plt.figure(num=3, figsize=(5, 3))
# # ax1 = FigureNum2.add_subplot(211)                             
# # ax1.plot(filtered_x_tof[InPoint:NoDatapoints,0],
# #          filtered_x_tof[InPoint:NoDatapoints,1],
# #          'r-')
# # #ax1.plot(filtered_x_stacked[0:NoDatapoints,0],'b-')
# # ax2 = FigureNum2.add_subplot(212)                             
# # ax2.plot(filtered_x_tof[InPoint:NoDatapoints,0],
# #          filtered_x_tof[InPoint:NoDatapoints,2],
# #          'b-')
# # #ax2.plot(filtered_x_stacked[0:NoDatapoints,1],'b-')
# # ax2.set_xlabel('Time (s)', fontsize=6)
# # ax1.set_ylabel('Displacement TOF1 (mm)', fontsize=6) 
# # ax2.set_ylabel('Displacement TOF2 (mm)', fontsize=6) 
# # plt.xticks(fontsize=6)
# # plt.yticks(fontsize=6)
# =============================================================================
# WebCam
# =============================================================================
path = GenPath+"/DataFiles/Data18_03_2020/WebCam/"
name = "Marker_detection"
Bolustype = "Dry"
QSR_DoubleLayer_WebCam = TOFandWebCam(path, "stent_P1537_P0_40mm", name)
QSR_DoubleLayer_WebCam.find_all()
QSR_DoubleLayer_WebCam.FileReading([0,2,1])

dict_keys2=list(QSR_DoubleLayer_WebCam.ArrayDict.keys())


x_webcam={'0.6':QSR_DoubleLayer_WebCam.ArrayDict[dict_keys2[0]],
            '0.8':QSR_DoubleLayer_WebCam.ArrayDict[dict_keys2[1]],
            '1.0':QSR_DoubleLayer_WebCam.ArrayDict[dict_keys2[2]]}


TOFandWebCam.DispPlotting(plt.figure(num=3, figsize=(5, 3)),
                          3,
                          x_webcam['0.6'],x_webcam['0.8'],x_webcam['1.0'],
                          window=[(500,1700),(500,1700),(1500,3500)],
                          label=['a','b','c']
                          )

## Apply filter
QSR_DoubleLayer_WebCam.ApplyFilter(taps)

filtered_x_webcam=QSR_DoubleLayer_WebCam.ArrayDictFiltered


# #############################################################################
# Fit IsotonicRegression and LinearRegression models

ir_x_webcam={'0.6':TOFandWebCam.Detrending_isotonicRegression(filtered_x_webcam[dict_keys2[0]]),
             '0.8':TOFandWebCam.Detrending_isotonicRegression(filtered_x_webcam[dict_keys2[1]]),
             '1.0':TOFandWebCam.Detrending_isotonicRegression(filtered_x_webcam[dict_keys2[2]])}


TOFandWebCam.DispPlotting(plt.figure(num=4, figsize=(5, 3)),
                          3,
                          ir_x_webcam['0.6'],ir_x_webcam['0.8'],ir_x_webcam['1.0'],
                          window=[(500,1700),(500,1700),(1500,3500)],
                          label=['a','b','c']
                          )


# FigureNum5 = plt.figure(num=5, figsize=(5, 3))
# TOFandWebCam.DispPlotting(FigureNum5,
#                           1,
#                           ir_x_webcam,
#                           window=(500,1700)
#                           )


# # filtered_x_webcam=np.zeros((x_webcam.shape))

# # filtered_x_webcam[:,0]=x_webcam[:,0]
# # filtered_x_webcam[:,1] = lfilter(taps, 1, x_webcam[:,1])


# # ##Plot
# # InPoint=600
# # NoDatapoints=2100                             
# # FigureNum4 = plt.figure(num=4, figsize=(5, 1.5))
# # ax1 = FigureNum4.add_subplot(111)                             
# # ax1.plot(filtered_x_webcam[InPoint:NoDatapoints,0]-filtered_x_webcam[InPoint,0],
# #          filtered_x_webcam[InPoint:NoDatapoints,1]-filtered_x_webcam[InPoint,1],
# #          'r-')
# # ax1.set_xlabel('Time (s)', fontsize=6)
# # ax1.set_ylabel('Displacement Marker (mm)', fontsize=6) 
# # plt.xticks(fontsize=6)
# # plt.yticks(fontsize=6)

# # =============================================================================
# # # Compute noisy derivative by applying total variation regularisation
# # =============================================================================

# DiffrentiationType='TVRegDiff'

# dx_TVRegDiff,x_aug_TVRegDiff=SINDYc_MPC_Design.ComputeDerivative(DiffrentiationType,
#                                                      x_QSR_DoubleLayer,
#                                                      u,
#                                                      dt)

# # =============================================================================
# # ## Comparing the clean and noisy data
# # =============================================================================

# f1 = plt.figure()
# ax1 = f1.add_subplot(111)

# a12=ax1.plot(tspan_QSR_DoubleLayer,
#              x_QSR_DoubleLayer[:,0],color='darkred',
#              #ls='--',
#                         alpha=1,
#                         linewidth=1)

# #a11=ax1.plot(tspan_QSR_DoubleLayer[50:-50],
# #             x_aug_TVRegDiff[:,0],color='k',
# #                        alpha=1,
# #                        linewidth=1)

# ax1.set_xlabel(r"time t", size=12)
# ax1.set_ylabel(r"${{x_1}}(t)$", size=12)
