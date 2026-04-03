#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:03:50 2020

@author: pi
"""

import spidev
import time

#spi2 = spidev.SpiDev()
#spi2.open(0, 1)
#spi2.max_speed_hz = 976000
#
#spi2.writebytes([int(0),int(DAC_DATA[0,0])])

class MCP3008:
    def __init__(self, bus = 0, device = 0):
        self.bus, self.device = bus, device
        self.spi = spidev.SpiDev()
        self.open()
 
    def open(self):
        self.spi.open(self.bus, self.device)
    
    def read(self, channel = 0):
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data
            
    def close(self):
        self.spi.close()
        
while True:
    try:
        adc = MCP3008()
        value = adc.read( channel = 0 ) # You can of course adapt the channel to be read out
        print("Applied voltage: %.2f" % (value / 1023.0 * 3.3) )
        time.sleep(1)
    except KeyboardInterrupt:
        break
        print('Bye bye')