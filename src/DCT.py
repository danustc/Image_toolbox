"""
Last update by Dan on 10/27/2016.
"""

import numpy as np
import scipy.fftpack as fftpack
import io 
import os

import matplotlib.pyplot as plt


# shared small functions 





def dct_2D(i_slice,inv = False):
    if inv == False:
    # image slice
        return fftpack.dct(fftpack.dct(i_slice.T, norm = 'ortho').T, norm = 'ortho')
    else:
        return fftpack.idct(fftpack.idct(i_slice.T, norm = 'ortho').T, norm = 'ortho')



def dct_coefficients(dct_mat):
    coef_ampli = dct_mat.ravel()
    coef_range = range(coef_ampli.size)
    
    return coef_range
    