'''
A collection of image tools on the Fourier domain.
'''
import numpy as np


def band_pass_dumb(NY, NX, k_low, k_high):
    '''
    the dumbest band pass filter.
    '''
    HX = NX//2
    HY = NY//2
    HR = np.min([HY,HX])
    MX, MY = np.meshgrid(np.arange(NX)-HX, np.arange(NY)-HY)
    RR = np.sqrt(MX**2 + MY**2)
    mask = np.logical_and(RR>k_low*HR, RR<k_high*HR)
    return mask

