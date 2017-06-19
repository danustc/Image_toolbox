'''
Removing high-frequency noises (those wrongly extracted cells)
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
import pyfftw
import scipy.fftpack as fftpack

def freq_cut(dff_raw, fcut, dt = 0.5):
    '''
    dff_raw: the raw Delta F/F signals, each column represents DF/F of a cell.
    fcut: the cutting frequency
    dt: time step
    return: the indices of the cells that show little activities.
    '''
    ft_dff = fftpack.fftshift(fftpack.fft(dff_raw, axis = 0)) # zero frequency at center 
    NT, NP = dff_raw.shape
    T = NT*dt # the total duration
    fNyq = 0.5/dt # Nyquist frequency
    df= 2*fNyq/NT # the frequency resolution
    Ncut = int(np.min([fNyq,fcut])/df) # where to cut off

    return ft_dff, Ncut





def group_denoise(dff_raw, fcut = 0.6, dt = 0.5, gvar = 0.95):
    '''
    0. perform Fourier transform on all the df/f and cut out the high-frequency components
    1. perform PCA and select the most
    '''
    pass
