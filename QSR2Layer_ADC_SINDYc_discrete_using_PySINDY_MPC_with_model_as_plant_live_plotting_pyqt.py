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
from QSR2Layer_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant import*

class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")
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
        self.xlim = 200
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
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0, 100)
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

    
    # Simulate some data
    n = np.linspace(0, 499, 500)
    y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
    i = 0
    QSR_DoubleLayer_Obj_SINDYc_SI=sindyc_model()
    
    # Choose prediction horizon over which the optimization is performed
    x0=np.array([100])#x0_robot_adc_valid
    Nvec=np.array([1])
    
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
        x0n=x0.T#[100; 50];             # Initial condition
        uopt0 = 70                     # Set initial control input to thirty
        
        # Constraints on control optimization
        LB = 100#[] #-100*ones(N,1);        # Lower bound of control input
        UB = 115           # Upper bound of control input
        
        # Reference state, which shall be achieved
        xref1 = np.array([75]).T
        
        x        = x0n
        x=np.asarray(x)
        
        Ton      = 30      # Time when control starts
        uopt     = uopt0*np.ones(N)
        uopt=np.asarray(uopt)
        
        xHistory = x       # Stores state history
        uHistory = uopt[0] # Stores control history
        tHistory = np.zeros((1))       # Stores time history
        rHistory = xref1   # Stores reference (could be trajectory and vary with time)
        
        #pdb.set_trace()
        
        bound_optVar=[(LB,UB)]
        
        funval=np.zeros((int(Duration/Ts)))
        
        # Setup the signal-slot mechanism.
        mySrc = Communicate()
        mySrc.data_signal.connect(addData_callbackFunc)
        
        mySrc1 = Communicate()
        mySrc1.data_signal.connect(addData_callbackFunc)
        
        for ct in range(int(Duration/Ts)):
                
                if ct*Ts>30:   # Turn control on
                    
                    if ct*Ts==Ton+Ts:
                        print('Start Control.')
                
                    # Set references
                    xref = np.asarray(xref1)
    #                pdb.set_trace()
                    
                    obj=QSR_DoubleLayer_Obj_SINDYc_SI
                    
    
                    
                    uopt=minimize(obj.RobotObjectiveFCN,
                                  uopt,
                                  method='SLSQP',
                                  args=(x[0],Ts,N,xref,uopt[0],np.diag(Q),R,Ru),
                                  tol=0.1,
                                  options={'ftol': 0.1, 'disp': False},
                                  # constraints=cons,
                                  bounds=[(LB,UB)]
                                  )
                    # pdb.set_trace()
                    
                    funval[ct]=uopt.fun
                    
                    if np.absolute(x[0])>700:
                        break
                    
                    uopt=uopt.x
                    
                else:    #% If control is off
                    
                    uopt=uopt0*np.ones((N))
                    xref=-1000*np.ones((1))
                    
                # Integrate system: Apply control & Step one timestep forward
                if x.ndim==0:
                        x = x[np.newaxis]    
                        
                x = QSR_DoubleLayer_Obj_SINDYc_SI.model.predict(x,uopt[0])
                
                t_History=tHistory[-1]+Ts/1
                xHistory=np.vstack((xHistory,x))
                uHistory=np.vstack((uHistory,uopt[0]))
                tHistory = np.vstack((tHistory,t_History))
                rHistory = np.vstack((rHistory,xref))
                
                # print(x[0])
                time.sleep(0.01)
                mySrc.data_signal.emit(xref) # <- Here you emit a signal!
                # mySrc1.data_signal.emit(xref)
                # mySrc.data_signal.emit(xref)

if __name__== '__main__':
    
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
    