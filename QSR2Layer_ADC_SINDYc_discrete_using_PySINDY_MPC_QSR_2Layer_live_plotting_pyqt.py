#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:32:18 2020

@author: dipankarbhattacharya
"""

###################################################################
#                                                                 #
#                    PLOT A LIVE GRAPH (PyQt5)                    #
#                  -----------------------------                  #
#            EMBED A MATPLOTLIB ANIMATION INSIDE YOUR             #
#            OWN GUI!                                             #
#                                                                 #
###################################################################

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import functools
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
from QSR2Layer_ADC_SINDYc_discrete_using_PySINDY_MPC_QSR_2Layer import*

class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("Two Layer QSR Output Window")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        # Place the zoom button
        self.zoomBtn = QPushButton(text = 'zoom')
        self.zoomBtn.setFixedSize(100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0,0))
        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        self.show()
        return

    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)
        return

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return

''' End Class '''


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        self.addedData = []
        print(matplotlib.__version__)
        # The data
        self.xlim = 1000
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50
        # The window
        self.fig = Figure(figsize=(5,5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('Discrete time steps')
        self.ax1.set_ylabel('Two layer QSR recorded pressure (kPa)')
        self.line1 = Line2D([], [], color='royalblue')
        self.line1_tail = Line2D([], [], color='maroon', linewidth=2)
        self.line1_head = Line2D([], [], color='seagreen', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, 1000)#self.xlim - 1)
        self.ax1.set_ylim(40, 80)
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])
        return

    def addData(self, value):
        self.addedData.append(value)
        return

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()
        return

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]
        return

''' End Class '''


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QObject):
    data_signal = pyqtSignal(float)

''' End Class '''

def dataSendLoop(addData_callbackFunc):

    try:
        # =============================================================================
        #Generate SindyC model and prediction
        # =============================================================================
        QSR_DoubleLayer_Obj_SINDYc_SI=sindyc_model()
        
        # =============================================================================
        # various prediction horizon loop
        # =============================================================================
        # Choose prediction horizon over which the optimization is performed
        x0=np.array([70])#x0_robot_adc_valid
        Nvec=np.array([4])
        
        for i in range(Nvec.shape[0]):
            
            Ts          = .1              # Sampling time
            N           = Nvec[i]          # Control / prediction horizon (number of iterations)
            Duration    = 100              # Run control for 100 time units
            Nvar        = 1
            Q           = np.array([1])            # State weights
            #Q=Q[np.newaxis,:]
            R           = 0.5 #0.5;         # Control variation du weights
            Ru = 0.5#0.01                 # Control weights
        #    B = np.array([0,1])                    # Control vector (which state is controlled)
        #    B=B[:,np.newaxis]
        #    #C = eye(Nvar)                  # Measurement matrix
        #    D = 0                          # Feedforward (none)
            x0n=x0.T                       # Initial condition
            uopt0 = 30                     # Set initial control input to thirty
            Ton      = 30                  # Time when control starts
            
            # Reference trajectory, which shall be achieved
            xref0 = 0*np.ones((1,int(Ton/Ts))).T #Initial part of the reference where the control is off
            xref1 = 60*np.ones((1,100)).T
            xref2 = 70*np.ones((1,250)).T
            xref4 = 75*np.ones((1,200)).T
            xref3 = 65*np.ones((1,151)).T
            
            args=(xref0,xref1,xref2,xref3,xref4)
            xref_all=np.concatenate(args)
            assert xref_all.shape[0]-1 == int(Duration/Ts), 'size of xref_all must be equal to int(Duration/Ts)'
            
            x        = x0n
            x=np.asarray(x)
            
            uopt     = uopt0*np.ones(N)
            uopt=np.asarray(uopt)
            
            xHistory = x       # Stores state history
            uHistory = uopt[0] # Stores control history
            tHistory = np.zeros((1))       # Stores time history
            rHistory = xref1   # Stores reference (could be trajectory and vary with time)
            x_Tof_History=np.zeros((3))
            
    
            
            funval=np.zeros((int(Duration/Ts)))
            
            start_time = time.time()
        
            # RoSE Actuation parameters
            BaseLinePress=np.zeros((1,12))
            ScalingFact=1.0
            num_rose_layers=12
                
            
            # Setup the signal-slot mechanism.
            mySrc = Communicate()
            mySrc.data_signal.connect(addData_callbackFunc)
            
            #Initialize TOF 
            t1 = timedelta(minutes = 0, seconds = 0, microseconds=0)
            
            if QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseADC is True and QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseIOExpander is True:
                
                TOFADCPer2dArray=np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).T
                range_mm_array=QSR_DoubleLayer_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)
    
                #Design real-time filter
                b,z=RoSE_actuation_protocol.Filter_RealTime_design(numtaps=50 )
                z=(z,z)
                TOF2dArray_mean_filtered_stacked=np.array([[],[],[]]).T
            
            #=============================================================================
            # MPC loop
            # =============================================================================
            
            for ct in range(int(Duration/Ts)):
                    
                    if ct*Ts>30:   # Turn control on
                        
                        if ct*Ts==Ton+Ts:
                            print('Start Control.')
                    
                        # Set references
                        #xref = np.asarray(xref1)
                        xref = xref_all[ct]
                        
                        # Constraints on control optimization
                        if xref==60:
                            LB=22.5 #Lower bound
                            UB=120  #Upper bound
                        elif xref==70:
                            LB=57.5
                            UB=120#77.5
                        elif xref==75:
                            LB=82.5
                            UB=120#92.5
                        elif xref==65:
                            LB=37.5
                            UB=120
                            
                        obj=QSR_DoubleLayer_Obj_SINDYc_SI
                        
        
                        
                        uopt=minimize(obj.RobotObjectiveFCN,
                                      uopt,
                                      method='SLSQP',
                                      args=(x[0],Ts,N,xref,uopt[0],np.diag(Q),R,Ru),
                                      tol=0.1,
                                      options={'ftol': 0.1, 'disp': False},
                                      # constraints=cons,
                                      bounds=[(LB,UB),(LB,UB),(LB,UB),(LB,UB)]
                                      )
                        # pdb.set_trace()
                        
                        funval[ct]=uopt.fun
                        
                        if np.absolute(x[0])>700:
                            break
                        
                        uopt=uopt.x
                        
                    else:    #% If control is off
                        
                        if ct*Ts==0:
                            print('Control is off')
                        
                        uopt=uopt0*np.ones((N))
                        xref=-1000*np.ones((1))
                        
                    # Integrate system: Apply control & Step one timestep forward
                    if x.ndim==0:
                            x = x[np.newaxis]   
                            
                    #Apply input to RoSE
                    uopt0_all_layers=uopt[0]*np.ones((1,num_rose_layers))
                    QSR_DoubleLayer_Obj_SINDYc_SI.mergeDACadd2DataAndSend(uopt0_all_layers,
                                                  0,
                                                  BaseLinePress,
                                                  ScalingFact,
                                                  num_rose_layers,0)
                    
                    #Generate ADC output from RoSE
                    if QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseIOExpander is True \
                    and QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseADC is True:
                        
                        
                        pressure_kpa_array=QSR_DoubleLayer_Obj_SINDYc_SI.GeneratePressureDataFromADC(NoOfADC=12)
                        
                        range_mm_array=QSR_DoubleLayer_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)
                        
                        range_mm_array_mean=np.array([range_mm_array[0],
                                                  range_mm_array[1:6].mean(axis=0),
                                                  range_mm_array[7:].mean(axis=0)])
                        range_mm_array_filtered, z= RoSE_actuation_protocol.Filter_RealTime_apply(range_mm_array_mean[1:],
                                                                                              b,z)
                        range_mm_array_filtered=range_mm_array_filtered[...,np.newaxis].T
                        range_mm_array_filtered=np.concatenate((np.array([range_mm_array[0]]),
                                                  range_mm_array_filtered[0]))
                        
    #                    TOFADCPer2dArray=np.concatenate((TOFADCPer2dArray,range_pressure_peristalsis_array[np.newaxis,:]))
                                            
                        
                    x = np.array([[pressure_kpa_array[3]]])
                                           
                    t_History=tHistory[-1]+Ts/1
                    xHistory=np.vstack((xHistory,x))
                    uHistory=np.vstack((uHistory,uopt[0]))
                    tHistory = np.vstack((tHistory,t_History))
                    rHistory = np.vstack((rHistory,xref))
                    
                    
                    # print(x[0])
#                    time.sleep(0.01)
                    mySrc.data_signal.emit(x[0]) # <- Here you emit a signal!
                    # mySrc1.data_signal.emit(xref)
                    # mySrc.data_signal.emit(xref)
                    
            # =============================================================================
            # Clear input to the RoSE:Actuate it with 0 kPa        
            # =============================================================================
            QSR_DoubleLayer_Obj_SINDYc_SI.RobotNoActuation()
            print('Cleared QSR/n')
                    
    except KeyboardInterrupt:
        Robot_clear=RoSE_actuation_protocol(UseIOExpander=False)
        ClearDAC=np.zeros((1,12),dtype=int)
        Robot_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)
#    except:
#        Robot_clear=RoSE_actuation_protocol(UseIOExpander=False)
#        ClearDAC=np.zeros((1,12),dtype=int)
#        Robot_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)

if __name__== '__main__':
    
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
    