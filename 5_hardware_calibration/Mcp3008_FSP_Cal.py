#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:00:55 2020

@author: pi
"""

import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
import pandas as pd
import numpy as np
from adafruit_mcp3xxx.analog_in import AnalogIn

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the cs (chip select)
cs = digitalio.DigitalInOut(board.D17)

# create the mcp object
mcp = MCP.MCP3008(spi, cs)

# create an analog input channel on pin 0
chan1 = AnalogIn(mcp, 0)
chan2 = AnalogIn(mcp, 1)

WtFSPArray=np.array([[],[]]).T
while True:
    try:
        wt_in_g=input('Enter weight in grams: ')
        num_of_sam=10
        while num_of_sam>0:
            print('Raw ADC Value: ', chan1.value,chan2.value)
            print('ADC Voltage: ' + str(chan1.voltage) + 'V','ADC Voltage: ' + str(chan2.voltage) + 'V')
            print(chan2.value/(chan1.value-chan2.value))
            FSP_val=chan2.value/(chan1.value-chan2.value)
            WtFSPArray=np.concatenate((WtFSPArray,np.array([[wt_in_g,FSP_val]])))
            time.sleep(1)
            num_of_sam=num_of_sam-1
            
    except ZeroDivisionError:
        continue
    except KeyboardInterrupt:
        filename='DataFiles/Data14_10_2020_FSP_Cal/FSP_cal_with_weight_test_5.csv'
        columns=('Wt in grams','A_-/(A_+ - A_-)')
        df_WtFSPArray=pd.DataFrame(WtFSPArray,columns=columns)
        df_WtFSPArray.to_csv(filename)
        print('Bye bye')
        break