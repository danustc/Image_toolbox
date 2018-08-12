'''
Cell segmentation algorithm based on the Cytometry paper.
Created by Dan on 08/09/2018.
Some thresholding functions to help segmentation.
'''
import numpy as np
from scipy.signal import argrelextrema, savgol_filter

def hist_group(img, nbins = 100, n_group = 2):
    '''
    img: raw image
    n_group: number of groups to be divided into
    '''

    hist, be = np.histogram(img, nbins)
    local_peaks = argrelextrema(hist, np.greater)[0] # local maxima
    local_valleys = argrelextrema(hist, np.less)[0] # local maxima



def local_thresholding(img, patchsize, stride = 100):
    '''
    img: raw image
    patchsize: the patch size (R, C), must be smaller than that of image size.
    '''
     

