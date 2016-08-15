'''
Created by Dan on 08/02/16
This file contains several small functions shared among all the classes.
Adapted from Scipy cookbook. 
'''

import numpy as np
from scipy import optimize

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
    integrate all the pixel values within a circle centered at cr with radius dr.
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
    

def groups_compare(dset1, dset2, tor = 1.0):
    """
    Comparing data set1 and data set2, yield the unison and differences 
    """
    
    