import numpy as np
import scipy as sp
from scipy.signal import convolve2d as conv2
from scipy import signal
from scipy import fft as pyfft
from skimage import color, data, restoration



# Create a gaussian 
def make_gaussian(xx, yy, width, center):
    x0 = center[0]
    y0 = center[1] # generate a meshgrid
    xv, yv = np.meshgrid(xx,yy)
    if len(width)==1: 
        dx = width
        dy = width
    else:
        dx=width[0]
        dy=width[1]
        
#     gm = np.zeros([len(yy), len(xx)]) # create an empty array
    efac = (xv-x0)**2/dx**2 + (yv-y0)**2/dy**2 # The exponents 
    gm = np.exp(-efac/2)
    return gm


def deconv_2d(image, psf2):
    decon = signal.deconvolve(image, psf2)

def deconv_3d(image, psf3):
    pass

    
