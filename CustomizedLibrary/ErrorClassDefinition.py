#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:01:25 2020

@author: pi
"""


class PeristalsisThresholdError(Exception): pass   

#Define an exception handling function for checking 'y' or 'n'
class YesNoError(Exception): pass  

#Define an exception handling function for wave speed checking
class SpeedError(Exception): pass

class SpeedMismatchError(Exception): pass