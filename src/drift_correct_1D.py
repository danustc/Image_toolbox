"""
This file tests drift correction in 1D instead of 2D. 
Created by Dan on 08/30/2016
Last update: 
"""

import numpy as np
from scipy.fftpack import fft

class Drift_correct_1D(object):
    def __init__(self, raw_stack):
        self.nz, self.ny, self.nx = raw_stack.shape
        self.raw_stack = raw_stack
        
    
    def drift_correct(self):
        stack = self.raw_stack
        sig_x = stack.sum(axis = 1)
        sig_y = stack.sum(axis = 2)
        
        return sig_x, sig_y
    
    
        
        