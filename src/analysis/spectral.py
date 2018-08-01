'''
Analysis in frequency domain
'''
import numpy as np
import pyfftw

def spectrogram(dff_r, dt, twindow):
    '''
    create spectrogram of the dff_r signal (single)
    '''
    NT = dff_r.size


def dff_frequency(dff_r, dt):
    '''
    Calculate the frequency representation of the dff.
    '''
    NT, NC = dff_r.shape
    dk = 1./(NT*dt)
    a = pyfftw.empty_aligned(NT, dtype = 'complex128')
    b = pyfftw.empty_aligned(NT, dtype = 'complex128')
    NK = int(NT/2)
    dff_k = np.empty((NK,NC))
    container = pyfftw.FFTW(a,b)
    for ic in range(NC):
        container(dff_r[:,ic])
        freq_cps = container.get_output_array()
        dff_k[:,ic] = np.abs(freq_cps[:NK])
    return dff_k, dk

