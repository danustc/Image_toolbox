'''
Created by Dan on 08/15/16
This file contains several small functions shared among all the classes.
This one has Numerical functions. For graphic functions, see graph_funcs.py
Adapted from Scipy cookbook.
Last update: 08/17/16 
'''

import numpy as np
from scipy import optimize
from scipy import fftpack 

def gaussian2D(height, center_x, center_y, width_x, width_y, ofst=0.):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)+ofst
                
                

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()-data.min()
    ofst = data.min()
    return height, x, y, width_x, width_y, ofst

def fitgaussian2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) -
                                 data)
    p = optimize.leastsq(errorfunction, params)[0]
    return p


def circ_mask(n_size, cr, dr):
    """
    apply a mask on the array to sum up 
    Update: no need to pass the whole arr_2d to the function. 
    """
    cy = cr[0]
    cx = cr[1]
    
    ly = n_size[0]
    lx = n_size[1]
    yg, xg = np.ogrid[-cy:ly-cy, -cx:lx-cx]
    mask = yg*yg+xg*xg <= dr*dr
    
    return mask
    # end of circ_mask
    
def circs_reconstruct(dims, blob_list):
    """
    Reconstruct an image (a 2d array) with a list of blobs marked on it.
    dims: (ny, nx)
    blob_list: a 2-d array  
    """
    new_frame = np.zeros(dims)
    for blob in blob_list:
        # [blob[0], blob[1], n_frame, dr, signal_int]
        cr = [blob[0], blob[1]] # the center of blob in the unit of pixel
        dr = blob[3]
        sigs = blob[-1]
        
        mask = circ_mask(dims, cr, dr)
        new_frame[mask] = sigs
        

    return new_frame 
    # done with circs_reconstruct
        

def rect_mask(n_size, c_nw, c_se):
    """
    Return indices of a rectangular mask on a 2d-array. 
    Input: northwest and southeast corner coordinates. 
    output: mask, boolean array
    Unfinished >_< 
    """

    ly = n_size[0]
    lx = n_size[1]
    
    mask = [ly,lx]
    
    return mask 


def freq_feature(arr, ax = 0, sfreq = 1.25):
    """
    Extract frequency features in the given arr. 
    Suppose the given array is 2-d.
    """
    ndims = np.array(arr.shape)
    ft_arr = fftpack.fft(arr, axis = ax)
    N = ndims[ax]
    kk = np.arange(N)*sfreq*0.5/N # the k-axis 
    
    
    