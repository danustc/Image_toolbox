"""
This file tests drift correction in 1D instead of 2D. 
Created by Dan on 08/30/2016
Last update: 
"""

import numpy as np
import scipy.fftpack as fftp

class DC_1D(object):
    """
    Drift-correct 1D based 
    """
    def __init__(self, raw_stack):
        print(raw_stack.shape)
        self.nz, self.ny, self.nx = raw_stack.shape
        self.raw_stack = raw_stack
        
    
    def drift_correct(self):
        stack = self.raw_stack
        sig_x = stack.sum(axis = 1)
        sig_y = stack.sum(axis = 2)
        
        self.sig_x = sig_x
        self.sig_y = sig_y
        return sig_x, sig_y
    
    def fft_pair(self):
        """
        self.signal
        """
        sig_x = self.sig_x
        fsig_x = fftp.fft(sig_x, n=None, axis = 1, overwrite_x = False)
    
        np.conj(fsig_x[1:])
        
        