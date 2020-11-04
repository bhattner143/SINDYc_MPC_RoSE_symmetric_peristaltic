#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:29:58 2020

@author: pi
"""

from micropython import const

import adafruit_bus_device.i2c_device as i2c_device
import pdb
import time
import board
import busio
import pickle
import time
import numpy as np
from datetime import timedelta,datetime
from digitalio import Direction, Pull
from adafruit_mcp230xx.mcp23017 import MCP23017
from CustomizedLibrary.Class_vl6180x import VL6180X

from CustomizedLibrary.Class_RoSE_actuation_protocol_for_cal import RoSE_actuation_protocol_for_cal
# Reg nam                                           Reg add            No of bits
_VL6180X_DEFAULT_I2C_ADDR                         = const(0x29) 
#Registers defined by dips
_VL6180X_REG_SYSRANGE_PART_TO_PART_RANGE_OFFSET   = const(0x024)         #8
_VL6180X_REG_SYSRANGE_CROSSTALK_COMPENSATION_RATE = const(0x01e)         #16
_VL6180X_REG_RESULT_RANGE_RETURN_RATE             = const(0x066)         #16
_VL6180X_SYSRANGE_RANGE_CHECK_ENABLES             = const(0x02d)         #8
_VL6180X_SYSRANGE_RANGE_IGNORE_THRESHOL           = const(0x026)         #16
_VL6180X_SYSRANGE_MAX_CONVERGENCE_TIME            = const(0x01c)         #8

class vl6180x_calibration(VL6180X):

    def __init__(self, i2c=None, new_address=_VL6180X_DEFAULT_I2C_ADDR):
        
                                             
        self.vl6180x_new_address=new_address
                
#        self._reset_TOF_address
        self.i2c=i2c
        super().__init__(self.i2c,
             address=_VL6180X_DEFAULT_I2C_ADDR ,
             new_address=self.vl6180x_new_address)
    
    
#    @property   
#    def _reset_TOF_address(self):
#        
#        self.port_a_pins[0].value = False
#        self.port_a_pins[0].value = True
     
    def _range_avg_range_and_signal_return_rate(self,No_of_measurements):
        return np.mean(np.array([[self.range,self._range_signal_return_rate()] \
                                  for i in range(0,No_of_measurements)]),axis=0)
    
    def _range_signal_return_rate(self):
        #signal return rate in Mcps
        return self._read_16(_VL6180X_REG_RESULT_RANGE_RETURN_RATE)/128
    
    def _range_offset_calibration(self):
        """
        This section describes a procedure for calibrating offset.
            1. Clear the system offset
            a) Write 0x00 to SYSRANGE__PART_TO_PART_RANGE_OFFSET {0x24}.
            2. Place a target at 50mm away from VL6180X.
            a) Using a target with 17% reflectance or higher is recommended.
            b) When calibrating with cover glass we recommended using a target with
            reflectance of 88% or higher to minimise error from cross talk, cross
            3. Collect a number of range measurements with the target in place and calculate mean
            of the range results.
            a) For a reliable measurement, take at least 10 measurements.
            4. Calculate the offset required:
            5. Apply offset:
            a) Write the calculated offset value to
            SYSRANGE__PART_TO_PART_RANGE_OFFSET {0x24}.
            
            Note: {0x24} is in 2s complement representation. For offset value 0 to127, write 0x00 to 0x7f.
            For offset value -1 to -128, write 0xff to 0x80, e.g -30 = 0xe2
        """
        self._write_8(_VL6180X_REG_SYSRANGE_PART_TO_PART_RANGE_OFFSET, 0x00)
        
#        pdb.set_trace()
        No_of_measurements=10
        self.range_avg_actual=self._range_avg_range_and_signal_return_rate(No_of_measurements)[0]#np.array([self.range for i in range(0,self.No_of_measurements)]).mean()
        self.range_offset_mm=int(round(50-self.range_avg_actual))
        #For negative offset
        if self.range_offset_mm<0:
            tc=(-1)*(255 + 1 -abs(self.range_offset_mm))
            self.range_offset_mm=abs(tc)&0xff
                
        self.range_offset_hex=hex(int(self.range_offset_mm))
        
        with open('/home/pi/Desktop/SINDYc_MPC/TOFCalFiles/range_offset_mm_all_tof_'+hex(self.vl6180x_new_address)+'.pickle', 'wb') as f:
            pickle.dump(self.range_offset_mm, f)
        
        self._write_8(_VL6180X_REG_SYSRANGE_PART_TO_PART_RANGE_OFFSET, self.range_offset_mm)

    def _range_crosstalk_calbration(self):
        
        """
            This section describes a procedure for calibrating system cross talk.
            1. Perform the offset calibration (recommended)
            a) See Section 4.1.1.
            Note: If the offset is incorrectly calibrated, cross talk calibration will be inaccurate.
            2. Place a dark target at 100mm away from VL6180X.
            a) Low reflectance target recommended, e.g. 3% target.
            3. Ensure SYSRANGE__CROSSTALK_COMPENSATION_RATE {0x1e} is set to 0.
            4. Collect a number of range measurements with the target in place and calculate mean
            of the range results and Return signal rate.
            a) For a reliable measurement, take at least 10 measurements.
            5. Calculate the cross talk:
            6. Apply offset:
            a) Write the calculated cross talk value to
            SYSRANGE__CROSSTALK_COMPENSATION_RATE {0x1e}.
            Note: {0x1e} is a 316-bit register in 9.7 format:
            For cross talk value of 0.4 Mcps = 0.4 x 128 =51.2, register value to be written = 0x33.
        """
        if self._read_16(_VL6180X_REG_SYSRANGE_CROSSTALK_COMPENSATION_RATE)!=0:
            self._write_16(_VL6180X_REG_SYSRANGE_CROSSTALK_COMPENSATION_RATE , 0x0000)
        
        No_of_measurements=10
        #self.range_avg_actual_crosstalk=
        range_avg_range_and_signal_return_rate=self._range_avg_range_and_signal_return_rate(No_of_measurements)
        
        avg_return_rate_in_mcps=range_avg_range_and_signal_return_rate[1]
        avg_range_in_mm=range_avg_range_and_signal_return_rate[0]
        
        self.range_cross_talk_in_mcps=1*(avg_return_rate_in_mcps*(1-(1/100)*avg_range_in_mm))
        
        with open('/home/pi/Desktop/SINDYc_MPC/TOFCalFiles/range_cross_talk_mcps_all_tof_'+hex(self.vl6180x_new_address)+'.pickle', 'wb') as f:
            pickle.dump(self.range_cross_talk_in_mcps, f)
#        print(self.range_cross_talk_in_mcps)
        
    def _range_set_max_convergence_time(self,_range_conv_time_in_bits=0x05):
        self._write_8(_VL6180X_SYSRANGE_MAX_CONVERGENCE_TIME,_range_conv_time_in_bits)
        
    def _range_ignore(self,
                      range_cross_talk_mcps):
        
        """
            The range ignore function in VL6180X can be enabled by setting bit 1 of
            SYSRANGE__RANGE_CHECK_ENABLES {0x2d}. If enabled, the ignore threshold must
            be specified.
            We recommend setting the ignore threshold to at least 1.2x cross talk.
            e.g. SYSRANGE__RANGE_IGNORE_THRESHOLD {0x26} = cross talk (Mcps) x 1.2
            A range ignore error will be flagged if the return signal rate is less than the ignore threshold.
        """
        
        self._write_8(_VL6180X_SYSRANGE_RANGE_CHECK_ENABLES ,0x02)
        
        ignore_threshold=int(round(1.2*range_cross_talk_mcps*128))
        self._write_16(_VL6180X_SYSRANGE_RANGE_IGNORE_THRESHOL ,ignore_threshold)
        
    def range_offset_cal_data_load(self):
        with open('/home/pi/Desktop/SINDYc_MPC_RoSE_symmetric_peristaltic/TOFCalFiles/range_offset_mm_all_tof_'+hex(self.vl6180x_new_address)+'.pickle', 'rb') as f:
            range_offset_mm = pickle.load(f)
#        print('Range offset',range_offset_mm)
        self._write_8(_VL6180X_REG_SYSRANGE_PART_TO_PART_RANGE_OFFSET, range_offset_mm)
        
    def range_crosstalk_cal_data_load(self,RangeIgnoreFlag=True):
        
        with open('/home/pi/Desktop/SINDYc_MPC_RoSE_symmetric_peristaltic/TOFCalFiles/range_cross_talk_mcps_all_tof_'+hex(self.vl6180x_new_address)+'.pickle', 'rb') as f:
            range_cross_talk_mcps=pickle.load(f)
        range_cross_talk_bits=int(round(range_cross_talk_mcps*128))
        self._write_16(_VL6180X_REG_SYSRANGE_CROSSTALK_COMPENSATION_RATE,range_cross_talk_bits)
        
        if RangeIgnoreFlag is True:
            #Range ignore
            self._range_ignore(range_cross_talk_mcps)
            
def InitializeIOExpander(i2c):
            
    mcp = MCP23017(i2c)
    No_MCP23017_pins=10
    port_a_pins = [mcp.get_pin(pin) for pin in range(0, No_MCP23017_pins)]
    
    # Set all the port A pins to output
    for pin in port_a_pins:
        pin.direction = Direction.OUTPUT
        
    return port_a_pins
# =============================================================================
# Mainf function       
# =============================================================================
if __name__ == '__main__':      
    try:
        
        RoSE_Tof_cal=RoSE_actuation_protocol_for_cal(UseADC=True)
        dfFlipped=10*np.ones((1,12))
        size_df=dfFlipped.shape
#        Peristalsis_filepath='PeristalsisData/40mmat20mmps_Dips.csv'
#        dfFlipped=RoSE_Tof_cal.PeristalsisFileRead(Peristalsis_filepath)
        
        
        # Initialize the I2C bus:
        i2c = busio.I2C(board.SCL, board.SDA)
        port_a_pins=InitializeIOExpander(i2c)
        
        port_a_pins[0].value = False
        port_a_pins[1].value = False
        port_a_pins[2].value = False
        port_a_pins[3].value = False
        port_a_pins[4].value = False
        port_a_pins[5].value = False        
        port_a_pins[6].value = False
        port_a_pins[7].value = False
        port_a_pins[8].value = False
        port_a_pins[9].value = False
        
                
        TOF_cal=[]
        new_address=(const(0x40),const(0x41),const(0x42),
                     const(0x43),const(0x44),const(0x45),
                     const(0x46),const(0x47),const(0x48),
                     const(0x49)) 
        
        RangeOffsetCalFlag=True
        RangeCrossTalkCalFlag=True
        No_of_tof=10#len(new_address)
        Tof_assess=1
        act_array=np.array([10,25,40])
        #Initialize all TOF instance as a list
        for ii in range (0,No_of_tof):
            port_a_pins[ii].value = True
            TOF_cal.append(vl6180x_calibration(i2c,
                                          new_address=new_address[ii]))
    
            print('TOF Befor Cal',str(ii+1),'-->',TOF_cal[ii]._range_avg_range_and_signal_return_rate(10),'\n')
#        TOF_cal=[vl6180x_calibration(new_address=new_address[ii]) for ii in range(0,len(new_address))]
        
            # Range offset calibration 
            if RangeOffsetCalFlag is True:
#                print(ii)
                
                
                if ii==Tof_assess-1:
                    BaseLinePress=act_array[0]*np.ones((1,size_df[1]),dtype=int)
                    RoSE_Tof_cal.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
#                    pdb.set_trace()
                    TOF_cal[ii]._range_offset_calibration()   
         
                TOF_cal[ii].range_offset_cal_data_load()
            
        
        
            # Range crosstalk calibration and range ignore
            if RangeCrossTalkCalFlag is True:
                
                if ii==Tof_assess-1:
                    BaseLinePress=act_array[2]*np.ones((1,size_df[1]),dtype=int)
                    RoSE_Tof_cal.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
                    time.sleep(10)
                    TOF_cal[ii]._range_crosstalk_calbration()    
                    
            
                TOF_cal[ii].range_crosstalk_cal_data_load()
    #    
    
        
        while True:
            
            try:
                #print(np.array([TOF_cal.range for i in range(0,10)]).mean())
#                t1 =time.time()
#                print(TOF_cal[0]._range_avg_range_and_signal_return_rate(5))
#                print('Range status',TOF_cal[0].range_status)
#                print('Time elapsed for TOF reading ',time.time()-t1)
                print('No actuation')
                BaseLinePress=act_array[0]*np.ones((1,size_df[1]),dtype=int)
                RoSE_Tof_cal.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
                
                time.sleep(10)
                
                for ii in range (0,No_of_tof):
                    print('TOF no actuation',str(ii+1),'-->',\
                          TOF_cal[ii]._range_avg_range_and_signal_return_rate(10))
    #            print('Return signal rate',TOF_cal._read_16(0x66),'\n',
    #                  #'Return convergence time',TOF_cal._read_8(0x7c),'\n',
    #                  'Return signal count',TOF_cal._read_16(0x06c))
                
                print('\n')
                
                print('Half actuation')
                BaseLinePress=act_array[1]*np.ones((1,size_df[1]),dtype=int)
                RoSE_Tof_cal.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
                
                time.sleep(10)
                
                for ii in range (0,No_of_tof):
                    print('TOF half actuation',str(ii+1),'-->',\
                          TOF_cal[ii]._range_avg_range_and_signal_return_rate(10))
                
                print('\n')
                
                print('Full actuation')
                BaseLinePress=act_array[2]*np.ones((1,size_df[1]),dtype=int)
                RoSE_Tof_cal.mergeDACadd2DataAndSend(dfFlipped,0,BaseLinePress)
                
                time.sleep(10)
                
                for ii in range (0,No_of_tof):
                    print('TOF full actuation',str(ii+1),'-->',\
                          TOF_cal[ii]._range_avg_range_and_signal_return_rate(10))
                    
                
                
            except OSError:
                print('Error occurred')
        
    except KeyboardInterrupt:
        port_a_pins[0].value = False
        port_a_pins[0].value = True
        RoSE_Tof_cal_clear=RoSE_actuation_protocol_for_cal()
        ClearDAC=np.zeros((1,12),dtype=int)
        RoSE_Tof_cal_clear.mergeDACadd2DataAndSend(ClearDAC,0,ClearDAC,0,12,0)
        
        print('Bye Bye')

    