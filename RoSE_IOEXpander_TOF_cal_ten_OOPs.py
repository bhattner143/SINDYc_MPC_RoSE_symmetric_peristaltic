#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:34:35 2020

@author: pi

"""

import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import findiff as fd
import itertools

import smbus
import spidev
import time
import sys
import board
import busio
import digitalio
import RPi.GPIO as GPIO
import adafruit_mcp3xxx.mcp3008 as MCP

from adafruit_mcp3xxx.analog_in import AnalogIn
from datetime import timedelta,datetime
from scipy.signal import lfilter, firwin, freqz
from scipy import zeros, signal, random
from digitalio import Direction, Pull
from adafruit_mcp230xx.mcp23017 import MCP23017
#from CustomizedLibrary.Class_vl6180x import VL6180X
from CustomizedLibrary.Class_vl6180x_calibration import vl6180x_calibration
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


class RoSE_actuation_protocol():
    
    def __init__(self,RPi_Cobbler_Pin_No=18,
                 SPI_args=((0,0),5000000,0,False),
                 I2C_args=(1,0x65),
                 UseIOExpander=False,
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
        self.Flag_UseIOExpander=None
        self.Flag_UseADC=None
        
        
        if UseIOExpander is True:
            self.Flag_UseIOExpander=True
            #Call IOExpander initialisation and TOF address change function
            self.InitializeIOExpanderAndChangeTOFAddress((0x40,0x41,0x42,0x43,0x44,\
                                                          0x45,0x46,0x47,0x48,0x49))
        else:
            self.Flag_UseIOExpander=False

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
        
    def InitializeIOExpanderAndChangeTOFAddress(self,
                                                *VL6180X_NEW_I2C_ADDR):
        #pdb.set_trace()
        
        
        #With RPi GPIO pin reset the IOExpander
        GPIO.output(self.RPi_Cobbler_Pin_No,GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.RPi_Cobbler_Pin_No,GPIO.HIGH)

        '''
        # Initialize the MCP23017 chip on the bonnet
        # Optionally change the address of the device if you set any of the A0, A1, A2
        # pins.  Specify the new address with a keyword parameter:
        #mcp = MCP23017(i2c, address=0x21)  # MCP23017 w/ A0 set
        '''
        time.sleep(1)
        
        # Initialize the I2C bus:
        i2c = busio.I2C(board.SCL, board.SDA)
        mcp = MCP23017(i2c)
        No_MCP23017_pins=12
        port_ab_pins = [mcp.get_pin(pin) for pin in range(0, No_MCP23017_pins)]
    
        
        
        # Set all the port A pins to output
        for pin in port_ab_pins:
            pin.direction = Direction.OUTPUT
            
        
        # If needed, define new addresses for the i2c as a tuple and Create sensor instance.
        VL6180X_NEW_I2C_ADDR=VL6180X_NEW_I2C_ADDR[0]
        
        Num_of_TOF=len(VL6180X_NEW_I2C_ADDR)
        port_ab_pins[0].value = False
        port_ab_pins[1].value = False
        port_ab_pins[2].value = False
        port_ab_pins[3].value = False
        port_ab_pins[4].value = False
        port_ab_pins[5].value = False        
        port_ab_pins[6].value = False
        port_ab_pins[7].value = False
        port_ab_pins[8].value = False
        port_ab_pins[9].value = False

        
        TOF_sensor=[]

        #Initialize all TOF instance as a list
        for ii in range (0,Num_of_TOF):
            port_ab_pins[ii].value = True
            TOF_sensor.append(vl6180x_calibration(i2c,
                                          new_address=VL6180X_NEW_I2C_ADDR[ii]))

            TOF_sensor[ii].range_offset_cal_data_load()
            TOF_sensor[ii].range_crosstalk_cal_data_load()
#            print(TOF_sensor[ii]._range_avg_range_and_signal_return_rate(5))
#            print('Range status',TOF_cal.range_status)
                              
            time.sleep(0.1)
        self.TOF_sensor=TOF_sensor
#==============================================================================
        # Define a function to generate timestamp       
#==============================================================================        

    def GenerateTimestamp(self,t1):
        
        t=datetime.time(datetime.now())
        t2 = timedelta(minutes = t.minute, seconds = t.second, microseconds=t.microsecond)
        t3 = t2 - t1
        return t3.total_seconds()
#==============================================================================
        # Define a function to command the TOF VL 6180x           
#==============================================================================
    def GenerateDisplacementDataFromTOF(self,t1):  
        return np.array([self.GenerateTimestamp(t1),
                         self.TOF_sensor[0].range,
                             #self.TOF_sensor[0].range,self.TOF_sensor[0].range,self.TOF_sensor[0].range,self.TOF_sensor[0].range,
                         self.TOF_sensor[1].range,
                             #self.TOF_sensor[1].range,self.TOF_sensor[1].range,self.TOF_sensor[1].range,self.TOF_sensor[1].range,
                         self.TOF_sensor[2].range,
                             #self.TOF_sensor[2].range,self.TOF_sensor[2].range,self.TOF_sensor[2].range,self.TOF_sensor[2].range,
                         self.TOF_sensor[3].range,
                             #self.TOF_sensor[3].range,self.TOF_sensor[3].range,self.TOF_sensor[3].range,self.TOF_sensor[3].range,
                         self.TOF_sensor[4].range,
                             #self.TOF_sensor[4].range,self.TOF_sensor[4].range,self.TOF_sensor[4].range,self.TOF_sensor[4].range,
                         self.TOF_sensor[5].range,
                             #self.TOF_sensor[5].range,self.TOF_sensor[5].range,self.TOF_sensor[5].range,self.TOF_sensor[5].range,
                         self.TOF_sensor[6].range,
                             #self.TOF_sensor[6].range,self.TOF_sensor[6].range,self.TOF_sensor[6].range,self.TOF_sensor[6].range,
                         self.TOF_sensor[7].range,
                             #self.TOF_sensor[7].range,self.TOF_sensor[7].range,self.TOF_sensor[7].range,self.TOF_sensor[7].range,
                         self.TOF_sensor[8].range,
                             #self.TOF_sensor[8].range,self.TOF_sensor[8].range,self.TOF_sensor[8].range,self.TOF_sensor[8].range,
                         self.TOF_sensor[9].range])
                             #self.TOF_sensor[9].range,self.TOF_sensor[9].range,self.TOF_sensor[9].range,self.TOF_sensor[9].range]
                             

            
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
    
    # =============================================================================
    #    ADC: MCP 3008 for FSP 
    # =============================================================================
    def InitializeADCForFSP(self):
        # create the spi bus
        spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
        
        # create the cs (chip select)
        cs = digitalio.DigitalInOut(board.D17)
        
        # create the mcp object
        mcp = MCP.MCP3008(spi, cs)
        # create an analog input channel on pin 0 and pin 1
        self.chanList=[AnalogIn(mcp, num_chan) for num_chan in range(0,2)]

#==============================================================================
#     
#==============================================================================
    def StringCheck(self):

        while True:
            try:
                startActuation =input("Do you want to start actuation? (y/n) ")
                if startActuation!='y':
                    raise YesNoError('Input expects y or n')
                break
            except YesNoError:
                print('Input expects y or n, please try again...')      
        return startActuation
#==============================================================================
#     
#==============================================================================
    def SpeedCheck(self,filename):

        while True:
            try:
               SpeedOfWave=int(input("Enter integer value of wave speed in mm/s (20,30,40): "))
               SpeedList=[20,30,40]
               if SpeedOfWave in SpeedList:
                    tempString=str(SpeedOfWave)
                    
                    if tempString in filename:
                        index=SpeedList.index(SpeedOfWave)
                        NumSamplesList=[10,6.5,3.0]
                        SamplingTime=15/(SpeedOfWave*NumSamplesList[index])
                        break
                    else:
                        raise SpeedMismatchError("The wave speed entered and specified in the peristalsis data file name does not match. ")
                 
               else:
                   raise SpeedError('Input expects 20mm/s, 30mm/s or 40mm/s')
            except (ValueError ,SpeedError):
                print('Input expects integer, and 20mm/s, 30mm/s or 40mm/s')
            except SpeedMismatchError:
                print('The wave speed entered and specified in the peristalsis data file name does not match. Please try again...')
                
        SpeedOfWaveChecked=SpeedOfWave
        return SpeedOfWaveChecked, SamplingTime

#==============================================================================
#     
#==============================================================================
    @staticmethod
    def Data_Saving(SavingData, filename,ScalingFactor,DataFrom='TOF'):

        if DataFrom=='TOF':
            filenameForSaving=filename[16:]
            
            ColumnName=('time (ms)',
                        'TOF1 mean displacement (mm)',
                        'TOF2 mean displacement (mm)',
    ##                    'Sample 1_3 (mm)',
    ##                    'Sample 1_4 (mm)',
    ##                    'Sample 1_5 (mm)',
    ##                    'Sample 2_1 (mm)',
    ##                    'Sample 2_2 (mm)',
    ##                    'Sample 2_3 (mm)',
    ##                    'Sample 2_4 (mm)',
    ##                    'Sample 2_5 (mm)'
                        )
            
            dfTOF=pd.DataFrame(SavingData,columns=ColumnName)
            filename2="CsvData18_03_2020/QSR_2Layer_"+DataFrom+"_test_"+str(ScalingFactor)+"_"+\
                       filenameForSaving[0:4]+"_"+filenameForSaving[6:12]+\
                       ".csv"
            dfTOF.to_csv(filename2)

        elif DataFrom=='OnlineFilter':

            filenameForSaving=filename[16:]
            
            ColumnName=('time (ms)',
                        'OnlineFilter1 displacement (mm)',
                        'OnlineFilter2 displacement (mm)',
                        )
            
            dfOnlineFilter=pd.DataFrame(SavingData,columns=ColumnName)
            filename2="CsvData18_06_2020/QSR_2Layer_"+DataFrom+"_test_PDMS_sealing"+str(ScalingFactor)+"_"+\
                       filenameForSaving[0:4]+"_"+filenameForSaving[6:12]
            dfOnlineFilter.to_csv(filename2)

        elif DataFrom=='TOFADCandPer':

            filenameForSaving=filename[16:]
            dfTOFADCandPer=pd.DataFrame(SavingData)
            filename2="CsvData29_09_2020/RoSEv2pt0_"+DataFrom+"_test_"+str(ScalingFactor)+"_"+\
                       filenameForSaving[0:]
                       #+filenameForSaving[6:12]+\
            dfTOFADCandPer.to_csv(filename2)

    @staticmethod
    def TOF_Plottting(fig, PlottingData,PlottingDataFiltered,NumOfDataPts=[],defSize=4):

        if NumOfDataPts==[]:
            NumOfDataPts=PlottingData.shape[0]
            
        

        InPoint=200
        Num_of_plot=10
        ax=[]
        for ii in range(1,Num_of_plot):
            
            ax.append(fig.add_subplot(Num_of_plot,1,ii))
            
            ax[ii-1].plot(
                #PlottingData[InPoint:NumOfDataPts,0],
                PlottingData[InPoint:NumOfDataPts,ii],
                    color='red',
                    alpha=1,
                    linewidth=1.5,
                    linestyle='--',
                    label='TOF1',
                    #marker='o',
                    #ms=1,
                    #mec='white'
                       )
#            ax[ii-1].plot(
#                #PlottingDataFiltered[InPoint:NumOfDataPts,0],
#                PlottingDataFiltered[InPoint:NumOfDataPts,ii],
#                    color='blue',
#                    alpha=1,
#                    linewidth=1.5,
#                    #label='TOF1_filtered',
#                    linestyle='-',
#                    #marker='o',
#                    #ms=1,
#                    #mec='white'
#                       )

        
#        ax1.set_ylabel('Displacement (mm)',fontsize=defSize)
#        ax1.set_xlabel('Samples',fontsize=defSize)
#        ax2.set_ylabel('Displacement (mm)',fontsize=defSize)
#        ax2.set_xlabel('Samples',fontsize=defSize)
        

##        ax1.set_ylim([27,34])
##        ax2.set_ylim([40,45])

#        ax1.legend(loc='upper right',fontsize=defSize)
#        handles, labels = ax1.get_legend_handles_labels()
#
#        ax2.legend(loc='upper right',fontsize=defSize)
#        handles, labels = ax2.get_legend_handles_labels()
        
        plt.show()

        
        
    @staticmethod
    def Filter_RealTime_design(numtaps=100,cutoff=0.005):
        # taps=100 worked good for Peristalsis_Staircase_60_100_130_20mmps.csv

        b = signal.firwin(numtaps, cutoff)
        z = signal.lfilter_zi(b, 1)
    
        return b,z

    @staticmethod
    def Filter_RealTime_apply(actual_data,b,z,size_data=[],n=10):


        if n==1:
            #pdb.set_trace()
            return actual_data, z
        else:
            if size_data==[]:
                filtered_data = zeros((1,
                                   actual_data.shape[0]))
                filtered_data[0,0],z1  = signal.lfilter(b, 1, [actual_data[0]], zi=z[0])
                filtered_data[0,1],z2  = signal.lfilter(b, 1, [actual_data[1]], zi=z[1])
                filtered_data[0,2],z3  = signal.lfilter(b, 1, [actual_data[2]], zi=z[2])
                filtered_data[0,3],z4  = signal.lfilter(b, 1, [actual_data[3]], zi=z[3])
                filtered_data[0,4],z5  = signal.lfilter(b, 1, [actual_data[4]], zi=z[4])
                filtered_data[0,5],z6  = signal.lfilter(b, 1, [actual_data[5]], zi=z[5])
                filtered_data[0,6],z7  = signal.lfilter(b, 1, [actual_data[6]], zi=z[6])
                filtered_data[0,7],z8  = signal.lfilter(b, 1, [actual_data[7]], zi=z[7])
                filtered_data[0,8],z9  = signal.lfilter(b, 1, [actual_data[8]], zi=z[8])
                filtered_data[0,9],z10 = signal.lfilter(b, 1, [actual_data[9]], zi=z[9])
                #print('\n filtering',filtered_data[0,0])

                filtered_data, z=RoSE_actuation_protocol.Filter_RealTime_apply(filtered_data[0,:],
                                                                               b,
                                                                               (z1,z2,z3,z4,z5,z6,z7,z8,z9,z10),
                                                                               size_data=[],
                                                                               n=n-1)
                return filtered_data, z
                
   
#==============================================================================
# Main Program        
#==============================================================================
#Main function
if __name__ == '__main__':
    
    try:
        #Apply the Baseline Pressure
        while True:
            try:
                temp =int(input("Enter the baseline pressure for the ESR "))
                
                if temp>120:
                    raise PeristalsisThresholdError
                break
            
            except ValueError:
                print("Oops!  Only integer values are accepted.  Try again...")
            except PeristalsisThresholdError:    
                print('Please enter a value below 120..Please try again..')
                
        QSR_Two_layer=RoSE_actuation_protocol(UseIOExpander=True,UseADC=True)
        Flag_UseFSP=True
        Peristalsis_filepath='PeristalsisData/40mmat20mmps_Dips.csv'
        #Peristalsis_filepath='PeristalsisData/Peristalsis_Staircase_50_10_130_20mmps.csv'
#        Peristalsis_filepath='PeristalsisData/Peristalsis_Staircase_60_100_130_20mmps.csv'
        dfFlipped=QSR_Two_layer.PeristalsisFileRead(Peristalsis_filepath)
        
        print('\n')
        size_df=dfFlipped.shape
        BaseLinePress=temp*np.ones((1,size_df[1]),dtype=int)
        ZeroArray=np.zeros((1,size_df[1]),dtype=int)
        QSR_Two_layer.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
        print('Please wait for 4 seconds...')
        print('\n')
        time.sleep(0)

        ## Checking whether user wants to do the actuation or not
        startActuation=QSR_Two_layer.StringCheck()
        print('\n')
        ## Checking correct wave speed and it is integer data type or not? 
        SpeedOfWaveChecked, SamplingTime=QSR_Two_layer.SpeedCheck(Peristalsis_filepath)
        print('\n')
        
        ScalingFact =float(input("What is the scaling factor you would like? (Enter between 0-1.5): "))
        j=0;
        numOfCyc=0;
                
        Adc2dArray=np.zeros(size_df[1])
        TOF2dArray=np.zeros(7)

        t1 = timedelta(minutes = 0, seconds = 0, microseconds=0)
        
        if QSR_Two_layer.Flag_UseIOExpander is True and QSR_Two_layer.Flag_UseADC is False:
            
            #TOF2dArray=np.zeros((1,QSR_Two_layer.GenerateDisplacementDataFromTOF(t1).shape[0]))
            TOF2dArray=np.array([[],[],[]]).T
            range_mm_array=QSR_Two_layer.GenerateDisplacementDataFromTOF(t1)

            #Design real-time filter
            b,z=RoSE_actuation_protocol.Filter_RealTime_design()
            z=(z,z)
            TOF2dArray_mean_filtered_stacked=np.array([[],[],[]]).T

        elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is False:
            ADC2dArray=np.array([[],[],[],[],[],[],[],[],[],[],[],[]]).T

        elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is True:
            TOFADCPer2dArray=np.array([[],\
                                       [],[],[],[],[],[],[],[],[],[],\
                                       [],[],[],[],[],[],[],[],[],[],\
                                       [],[],[],[],[],[],[],[],[],[],[],[],\
                                       [],[],[],[],[],[],[],[],[],[],[],[]]).T
            range_mm_array=QSR_Two_layer.GenerateDisplacementDataFromTOF(t1)

            #Design real-time filter
            b,z=RoSE_actuation_protocol.Filter_RealTime_design()
            z=(z,z,z,z,z,z,z,z,z,z)
            TOF2dArray_mean_filtered_stacked=np.array([[],[],[],[],[],[],[],[],[],[],[]]).T
            
            if Flag_UseFSP is True:
                    QSR_Two_layer.InitializeADCForFSP()
                    FSPADC2dArray=np.array([[],[],[]]).T
        
        starttime=time.time()
        #==============================================================================
        # Looping starts
        #==============================================================================
        while True:
            
            # create the hex version of df
##            dfHex=[hex(df[j,x]) for x in range(12)]

            dac_data_array=QSR_Two_layer.mergeDACadd2DataAndSend(dfFlipped,j,BaseLinePress,ScalingFact,size_df[1],40)

            if QSR_Two_layer.Flag_UseIOExpander is True and QSR_Two_layer.Flag_UseADC is False:
                
                range_mm_array=QSR_Two_layer.GenerateDisplacementDataFromTOF(t1)
                range_mm_array_mean=np.array([range_mm_array[0],
                                              range_mm_array[1:6].mean(axis=0),
                                              range_mm_array[7:].mean(axis=0)])
                
                TOF2dArray=np.concatenate((TOF2dArray,range_mm_array_mean[np.newaxis,:]))

                range_mm_array_filtered, z= RoSE_actuation_protocol.Filter_RealTime_apply(range_mm_array_mean[1:],
                                                                                          b,z)
#                print(range_mm_array_filtered.shape)
                range_mm_array_filtered=np.concatenate((np.array([range_mm_array[0]]),
                                              range_mm_array_filtered[0,:]))
                TOF2dArray_mean_filtered_stacked=np.vstack((TOF2dArray_mean_filtered_stacked,range_mm_array_filtered
                                                          ))

            elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is False:
                pressure_kpa_array=QSR_Two_layer.GeneratePressureDataFromADC(NoOfADC=12)
                ADC2dArray=np.concatenate((ADC2dArray,pressure_kpa_array[np.newaxis,:]))

            elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is True:
                
#                t_tof_1 = time.time()
                range_mm_array=QSR_Two_layer.GenerateDisplacementDataFromTOF(t1)
#                print('Time elapsed for TOF rading ',time.time()-t_tof_1)
                pressure_kpa_array=QSR_Two_layer.GeneratePressureDataFromADC(NoOfADC=12)
                
                range_mm_array_mean=range_mm_array#np.array([range_mm_array[0],
#                                              range_mm_array[1:  6].mean(axis=0),
#                                              range_mm_array[6: 11].mean(axis=0),
#                                              range_mm_array[11:16].mean(axis=0),
#                                              range_mm_array[16:21].mean(axis=0),
#                                              range_mm_array[21:26].mean(axis=0),
#                                              range_mm_array[26:31].mean(axis=0),
#                                              range_mm_array[31:36].mean(axis=0),
#                                              range_mm_array[36:41].mean(axis=0),
#                                              range_mm_array[41:46].mean(axis=0),
#                                              range_mm_array[46:51].mean(axis=0)])
#                t_fil1=time.time()
                range_mm_array_filtered, z= RoSE_actuation_protocol.Filter_RealTime_apply(range_mm_array_mean[1:],
                                                                                          b,z)
#                print('Time elapsed for filteration ',time.time()-t_fil1)
                
                range_mm_array_filtered=range_mm_array_filtered[...,np.newaxis].T
                range_mm_array_filtered=np.concatenate((np.array([range_mm_array[0]]),
                                              range_mm_array_filtered[0]))
                
                range_pressure_array=np.hstack((np.hstack((range_mm_array_mean,
                           range_mm_array_filtered[1:])),
                           pressure_kpa_array))
                range_pressure_peristalsis_array=np.hstack((range_pressure_array,dac_data_array[0,:]))
                TOFADCPer2dArray=np.concatenate((TOFADCPer2dArray,range_pressure_peristalsis_array[np.newaxis,:]))
                
                if Flag_UseFSP is True:
                    try:
                        FSPADC2dArray=np.concatenate((FSPADC2dArray,np.array([[range_mm_array[0],
                                                                               QSR_Two_layer.chanList[0].value,
                                                                               QSR_Two_layer.chanList[1].value]])))
                    except ZeroDivisionError:
                        FSPADC2dArray=np.concatenate((FSPADC2dArray,np.array([[range_mm_array[0],
                                                                               0,
                                                                               0]])))
                

            
            QSR_Two_layer.GenerateTimestamp(t1)
            
            #time.sleep(SamplingTime- ((time.time() - starttime) % SamplingTime))
#            time.sleep(0.01)
            j+=1
            #if the row index of df reaches the end then go back to the starting point
            if j==dfFlipped.shape[0]: 
                numOfCyc+=1
                j=0
                print('\n')
                print('End of peristalsis cycle number: {0}'.format(numOfCyc))
                print('\n'*4)
                print('2.5s wait before initiating TOF reading...')
                #time.sleep(2.5 - ((time.time() - starttime) % 1))
                time.sleep(2.5)
                
    except KeyboardInterrupt:

        QSR_Two_layer_clear=RoSE_actuation_protocol(UseIOExpander=False)
        ClearDAC=np.zeros((1,12),dtype=int)
        QSR_Two_layer_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)

    # Generate a csv file from the TOF 2d data
        if QSR_Two_layer.Flag_UseIOExpander is True and QSR_Two_layer.Flag_UseADC is False:


            #Plot figure                                    
            Figure1= plt.figure()
            RoSE_actuation_protocol.TOF_Plottting(Figure1, TOF2dArray,TOF2dArray_mean_filtered_stacked)

            #Save Data
            TOF2dArray[:,0]=TOF2dArray[:,0]-TOF2dArray[0,0]
            TOF2dArray_mean_filtered_stacked[:,0]=TOF2dArray_mean_filtered_stacked[:,0]-\
                                                   TOF2dArray_mean_filtered_stacked[0,0]
            

            RoSE_actuation_protocol.Data_Saving(TOF2dArray,
                                               filename=Peristalsis_filepath,
                                               ScalingFactor=ScalingFact,
                                               DataFrom='TOF'
                                               )
            
            RoSE_actuation_protocol.Data_Saving(TOF2dArray_mean_filtered_stacked,
                                               filename=Peristalsis_filepath,
                                               ScalingFactor=ScalingFact,
                                               DataFrom='OnlineFilter'
                                               )
        elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is False:

            RoSE_actuation_protocol.Data_Saving(TOF2dArray,
                                           filename=Peristalsis_filepath,
                                           ScalingFactor=ScalingFact,
                                           DataFrom='TOFandADC'
                                           )
            
        elif QSR_Two_layer.Flag_UseADC is True and QSR_Two_layer.Flag_UseIOExpander is True:

            #Plot figure                                    
            Figure1= plt.figure()
            
            RoSE_actuation_protocol.TOF_Plottting(Figure1, TOFADCPer2dArray[:,[0,1,2,3,4,5,6,7,8,9,10]],TOFADCPer2dArray[:,[0,7,8,9,10,11,12,13,14,15,16]])
            
            RoSE_actuation_protocol.Data_Saving(TOFADCPer2dArray,
                                           filename=Peristalsis_filepath,
                                           ScalingFactor=ScalingFact,
                                           DataFrom='TOFADCandPer'
                                           )
            if Flag_UseFSP is True:
                # Save time-series tracked marker data
                ColumnName=('time', 'A0','A1')
                df_FSPADC2dArray=pd.DataFrame( FSPADC2dArray,columns=ColumnName)
                filename="CsvData29_09_2020/RoSEv2pt0_test_FSP_"#+str(ScalingFactor)+"_"+\
                       #filenameForSaving[0:]
                df_FSPADC2dArray.to_csv(filename)
                
                
        print('Bye! Bye! Dipu')
    except ValueError as ValErr:
        print(ValErr)
        print('Error...Check I2C Connection')
    except OSError as OSErr:
        print('% s occurred due to ADC not found at 65H' %OSErr)
    except IndexError as IndexErr:
        print(IndexErr)    
    finally:
        QSR_Two_layer_clear=RoSE_actuation_protocol(UseIOExpander=False)
        ClearDAC=np.zeros((1,12),dtype=int)
        QSR_Two_layer_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)
        print('All the best for next run')

        i2c = busio.I2C(board.SCL, board.SDA)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18,GPIO.OUT)        
        #With RPi GPIO pin reset the IOExpander
        GPIO.output(18,GPIO.LOW)
        time.sleep(1)
        GPIO.output(18,GPIO.HIGH)

        '''
        # Initialize the MCP23017 chip on the bonnet
        # Optionally change the address of the device if you set any of the A0, A1, A2
        # pins.  Specify the new address with a keyword parameter:
        #mcp = MCP23017(i2c, address=0x21)  # MCP23017 w/ A0 set
        '''
        time.sleep(1)
        i2c = busio.I2C(board.SCL, board.SDA)
        mcp = MCP23017(i2c)
        No_MCP23017_pins=12
        port_ab_pins = [mcp.get_pin(pin) for pin in range(0, No_MCP23017_pins)]


        # Set all the port A pins to output
        for pin in port_ab_pins:
            pin.direction = Direction.OUTPUT

        port_ab_pins[0].value = False 
        port_ab_pins[1].value = False
        port_ab_pins[2].value = False
        port_ab_pins[3].value = False
        port_ab_pins[4].value = False
        port_ab_pins[5].value = False        
        port_ab_pins[6].value = False
        port_ab_pins[7].value = False
        port_ab_pins[8].value = False
        port_ab_pins[9].value = False
        
        port_ab_pins[0].value = True
        port_ab_pins[1].value = True
        port_ab_pins[2].value = True
        port_ab_pins[3].value = True
        port_ab_pins[4].value = True
        port_ab_pins[5].value = True        
        port_ab_pins[6].value = True
        port_ab_pins[7].value = True
        port_ab_pins[8].value = True
        port_ab_pins[9].value = True
        
else:
    print("This module has been called from other program")
                
