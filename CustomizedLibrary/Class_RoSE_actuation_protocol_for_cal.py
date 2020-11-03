#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:02:29 2020

@author: pi
"""

import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import findiff as fd

import smbus
import spidev
import time
import sys
import board
import busio
import RPi.GPIO as GPIO

from datetime import timedelta,datetime
from scipy.signal import lfilter, firwin, freqz
from scipy import zeros, signal, random
from digitalio import Direction, Pull
from adafruit_mcp230xx.mcp23017 import MCP23017
#from CustomizedLibrary.Class_vl6180x import VL6180X
#from CustomizedLibrary.Class_vl6180x_calibration import vl6180x_calibration
from CustomizedLibrary.ErrorClassDefinition import PeristalsisThresholdError,YesNoError,SpeedError,SpeedMismatchError



# Clear shell
def cls():
    print('\n'*50)
#clear Console    
cls()

plt.style.use('dark_background')
plt.rcParams['figure.figsize']=(4.0,2.5 )
plt.rcParams['figure.dpi']=300
plt.rcParams['grid.linewidth']=0.5


class RoSE_actuation_protocol_for_cal():
    
    def __init__(self,RPi_Cobbler_Pin_No=18,
                 SPI_args=((0,0),5000000,0,False),
                 I2C_args=(1,0x65),
                 UseADC=False
                 ):
        # GPIO setup
        self.RPi_Cobbler_Pin_No= RPi_Cobbler_Pin_No
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RPi_Cobbler_Pin_No,GPIO.OUT)
        
        # SPI setup
        a,b,c,d=SPI_args
        a1,a2=a
        self.spi = spidev.SpiDev()
        self.spi.open(a1,a2)
        self.spi.max_speed_hz=b
        self.spi.mode=c
        #self.spi.cshigh=d

        # Flag setup
        self.Flag_UseADC=None
        

        if UseADC is True:
            self.Flag_UseADC=True
            
            #I2C setup for ADC
            # Define Smbus Configuration for ADC MAXIM 11605
            e,f=I2C_args
            ADC_BUS=e
            self.ADC_ADDRESS=f #7 bit address (will be left shifted to add the read write bit)
            #SEtup SMBus access
            self.ADCbus = smbus.SMBus(ADC_BUS)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
        else:
            self.Flag_UseADC=False

    def PeristalsisFileRead(self,Peristalsis_filepath='PeristalsisData/40mmat20mmps_Dips.csv' ):
        # Assign the spreadsheet filename to string filename
        #Load the spreadsheet
        df=pd.read_csv(Peristalsis_filepath)
        #convert dataframe array to int64 array
        df=np.array(df)
        #flip the matrix about the columns
        dfFlipped=np.flip(df,1)
        self.dfFlipped=dfFlipped
        
        return dfFlipped
        #size_df=self.df.shape

        # Define a function to command the ADC MAX11605
    def GeneratePressureDataFromADC(self,NoOfADC=12):

        if self.Flag_UseADC is True:
            """   
            With RPi GPIO pin reset the IOExpander
            GPIO.output(self.RPi_Cobbler_Pin_No,GPIO.LOW)

    
            This reads a single byte from a device, from a designated register.
            The register is specified through the Comm byte.
            S Addr Wr [A] Comm [A] S Addr Rd [A] [Data] NA P
            List comprehension to generate Data from all the ADCs at once

            Setup Byte
            REG=1, SEL2=1,SEL1=SEL0=0,CLK=0 (internal),BIP=0,RSTbar=1,X=0
#            SETUP_BYTE=1<<7|4<<4|2
#            self.ADCbus.write_byte_data(self.ADC_ADDRESS,SETUP_BYTE)
            
            Configuration byte
            REG=0, SCAN0=SCAN1=1,CS0to 3=ADC select, SGL=1
            """
#            i2c_adc = busio.I2C(board.SCL, board.SDA)
##            adafruit_vl6180x.VL6180X(i2c_adc,
##                                          address=self.ADC_ADDRESS
##                                          )
##            self._device = i2c_device.I2CDevice(i2c_adc, self.ADC_ADDRESS)

            pressure_kpa_array=np.array([self.ADCbus.read_byte_data(self.ADC_ADDRESS,3<<5|i<<1|1) for i in range(NoOfADC)])
##            self.ADCbus.write_byte_data(self.ADC_ADDRESS,SETUP_BYTE)
            
            return pressure_kpa_array
        
        elif self.Flag_UseADC is None:
            print("Please initialize the ADC when calling the constructor")
        

#==============================================================================
        # Define a function to generate timestamp       
#==============================================================================        

    def GenerateTimestamp(self,t1):
        
        t=datetime.time(datetime.now())
        t2 = timedelta(minutes = t.minute, seconds = t.second, microseconds=t.microsecond)
        t3 = t2 - t1
        return t3.total_seconds()
            
    def mergeDACadd2DataAndSend(self,dfMatrix,rowIndex,threshold,ScalingFact=0,size=12,Remove=40):
    #Loop through the columns of jth row of df
                 
                #raise ValueError        
        data12BitArray=np.zeros([size,1])
        temp=dfMatrix[rowIndex,:]
        if any(dfMatrix[rowIndex,:]-Remove)<0:
            temp[(dfMatrix[rowIndex,:]-Remove)<0]=ScalingFact*0+threshold
        else:
            temp=ScalingFact*((dfMatrix[rowIndex,:]-Remove))+threshold
    #==============================================================================
    #     # If any data value is greater than cut_off then we are passing (0x00ff & 296)=40 to DAC
    #==============================================================================
        cut_off=250
    #converting dac address and data to unsigned 8 bit
        data8Bit=0x00ff & 296 #=40
        DAC_DATA=int(data8Bit)
    #==============================================================================
    #         #Write the serial data to the ith dac
    #==============================================================================
        #resp=[self.spi.writebytes([ii,DAC_DATA]) for ii in range(0,size) if dfMatrix[rowIndex,ii]]        
        if any(dfMatrix[rowIndex,:]>cut_off) is True:
            resp=[self.spi.writebytes([ii,DAC_DATA]) for ii in range(0,size) if dfMatrix[rowIndex,ii]]        
        else:
            data8Bit=(0x00ff & temp.astype(np.uint8))
                        
            #converting dac address and data to unsigned 8 bit
            DAC_DATA=data8Bit.astype(np.uint8)
            data12BitArray=np.array([0,1,2,3,4,5,6,7,8,9,10,11],dtype=np.int16)
            #Merge the address of Dac with the 8 bit serial data to form 12 bit 
            data12BitArray=data12BitArray<<8|data8Bit
            #Write the serial data to the ith dac
            resp = self.spi.writebytes([int(0),int(DAC_DATA[0,0])])
            resp = self.spi.writebytes([int(1),int(DAC_DATA[0,1])])
            resp = self.spi.writebytes([int(2),int(DAC_DATA[0,2])])
            resp = self.spi.writebytes([int(3),int(DAC_DATA[0,3])]) 
            resp = self.spi.writebytes([int(4),int(DAC_DATA[0,4])])
            resp = self.spi.writebytes([int(5),int(DAC_DATA[0,5])])
            resp = self.spi.writebytes([int(6),int(DAC_DATA[0,6])])
            resp = self.spi.writebytes([int(7),int(DAC_DATA[0,7])])
            resp = self.spi.writebytes([int(8),int(DAC_DATA[0,8])])
            resp = self.spi.writebytes([int(9),int(DAC_DATA[0,9])])
            resp = self.spi.writebytes([int(10),int(DAC_DATA[0,10])])
            resp = self.spi.writebytes([int(11),int(DAC_DATA[0,11])])
            #pdb.set_trace()
        return DAC_DATA