'''
Cell segmentation algorithm based on the Cytometry paper.
Created by Dan on 08/09/2018.
Some thresholding functions to help segmentation.
'''
import numpy as np
from scipy.signal import argrelextrema, savgol_filter

def hist_group(img, nbins = 100, power_cut = 0.95, lower_cut = 50):
    '''
    img: raw image
    power_cut: the cutoff of pixel intensity.
    OK this is working.
    '''
    hist_init, be_init = np.histogram(img, nbins, density = True)
    db = be_init[1]-be_init[0]
    cs_hist = hist_init.cumsum()*db
    ind_cut = np.searchsorted(cs_hist, power_cut)
    # the second round of histogram  

    val_cut = be_init[ind_cut] # the upper bound of hist range
    hist_fine, be_fine = np.histogram(img, nbins, range = (lower_cut, val_cut))
    nw = int(nbins/20)*2+1 # must be an odd number
    hat = savgol_filter(hist_fine, nw, 3)
    local_peaks = argrelextrema(hat, np.greater)[0] # local maxima
    local_valleys = argrelextrema(hat, np.less)[0] # local maxima
    peaks = (be_fine[local_peaks]+be_fine[local_peaks+1])*0.50
    valleys = (be_fine[local_valleys]+be_fine[local_valleys+1])*0.50
    return peaks, valleys # the positions of peaks and valleys


def local_thresholding(img, patchsize, stride = 100):
    '''
    img: raw image
    patchsize: the patch size (R, C), must be smaller than that of image size.
    return: a matrix (ny_stride, nx_stride) storing local threshhold values.
    '''
    NY, NX = img.shape
    SY, SX = patchsize
    ny_stride = int((NY-SY) // stride)
    nx_stride = int((NX-SX) // stride)
    local_thresh = np.zeros((ny_stride, nx_stride))

    for ii in range(ny_stride):
        y_range = np.arange(SY)+ii*stride
        for jj in range(nx_stride):
            x_range = np.arange(SX) + jj*stride
            patch = img[y_range, x_range]
            lb = np.max([patch.min(), 50])
            peaks, valleys = hist_group(patch,  nbins = 100, power_cut = 0.95, param_lower_cut = lb)
            ind_val = np.searchsorted(peaks, valleys)
            local_thresh[ii,jj] = valleys[ind_val ==1][0] # take out only one element.

    return local_thresh
