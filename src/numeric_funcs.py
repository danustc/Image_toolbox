'''
Created by Dan on 08/15/16
This file contains several small functions shared among all the classes.
This one has Numerical functions. For graphic functions, see graph_funcs.py
Adapted from Scipy cookbook.
Last update: 09/13/16
'''

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import optimize
from scipy.ndimage import gaussian_filter1d



def smooth_lpf(shit_data, ft_width = 3):
    """
    shit_data: the noisy data that should be smoothed.
    ft_width: The filter width
    """
    s_filt = gaussian_filter1d(shit_data, ft_width, axis = 0)
    sca = np.min(shit_data/s_filt)
    s_final = shit_data - sca*s_filt

    return s_final, s_filt
    # done with smooth_lpf



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

    ind_mask = np.where(mask)
    return ind_mask # instead of returning a huge boolean matrix, just return a few indices
# -------------done with circ_mask


def circ_mask_patch(frame_size, cr, dr):
    """
    pass the test for 10x10, 120x 120 arrays.
    but failed for
    To find the mask of a small patch on a big frame.
    frame_size: ny, nx
    patch_size: scalar, at least 2* dr
    OK finally this is debugged.
    """

    c_lbound = np.floor(np.array(cr) - dr).astype('int16')  # Where the patch is cut off.
    c_ubound = np.floor(np.array(cr) + dr).astype('int16') +1
    c_patch = cr - c_lbound
    f_patch = c_ubound - c_lbound
    ind_patch = circ_mask(f_patch, c_patch, dr)

    if(ind_patch[0].size == 0):
        return ind_patch
    else:
        my = ind_patch[0] + c_lbound[0]
        mx = ind_patch[1] + c_lbound[1]
        msel_1 = np.logical_and(my >=0, mx>=0)
        msel_2 = np.logical_and(my< frame_size[0], mx < frame_size[1])
        msel = np.logical_and(msel_1, msel_2)
        ind_mask = (my[msel],mx[msel])
        return ind_mask

    # end of circ_mask

def circs_reconstruct(dims, blob_list, dr = 3):
    """
    Reconstruct an image (a 2d array) with a list of blobs marked on it.
    dims: (ny, nx)
    blob_list: a 2-d array, the first two columns specify the position of blobs in pixels, the third column specifies the signal intensity.
    This should be updated.

    """
    new_frame = np.zeros(dims)
    for blob in blob_list:
        # [blob[0], blob[1], n_frame, dr, signal_int]
        cr = [blob[0], blob[1]] # the center of blob in the unit of pixel
        sigs = blob[2]

        mask = circ_mask_patch(dims, cr, dr)
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
    # done with rect _mask


def corr_mat(arr_a, arr_b=None, scorr = False):
    """
    Convert two arrays, arr_a and arr_b into pandas dataframes and calculate the cross-correlations
    preserve the self-correlated part if scorr is True.
    This contains some redundant calculation if the self-calculation is discarded. However, iteration would be even slower.
    """
    if arr_b is None:
        # calculate self correlation
        df_a = df(arr_a)
        dcm = df_a.corr().as_matrix()

    else:
        na = arr_a.shape[1]
        nb = arr_b.shape[1]

        col_a = ['a'+str(x) for x in np.arange(na)]
        col_b = ['b'+str(x) for x in np.arange(nb)]
        df_a = df(arr_a, columns = col_a)
        df_b = df(arr_b, columns = col_b)

        df_concat = pd.concat([df_a,df_b], axis = 1)
        if scorr:
            dcc = df_concat.corr()
        else:
            dcc = df_concat.corr().ix[0:na, na:] # only taking out the upper right block, with na columns and nb rows.

        dcm = dcc.as_matrix()
    return dcm
    # done with corr_mat


def histo_peak(im_arr, val_cut, nbin = 50, ext = 1):
    """
    Construct a histogram of an image.
    cut off values below val_cut.
    Return peak position
    """

    hist, bdge = np.histogram(im_arr, bins = nbin)
    n_cut = np.searchsorted(bdge, val_cut)

    pmx = np.argmax(hist[n_cut:]) + n_cut

    sub_vals = bdge[pmx-ext:pmx+ext+1]
    sub_hist = hist[pmx-ext:pmx+ext+1]

    pk = np.inner(sub_vals, sub_hist)/sub_hist.sum()

#     pk = (bdge[pmx] + bdge[pmx+1])*0.5

    return pk
    # done with histo_peak
    
    
def lateral_distance(coord_1, coord_2):
    """
    Very simple: calculate distances between two groups of coordinates 
    """
    y1 = coord_1[:,0]
    y2 = coord_2[:,0]
    x1 = coord_1[:,1]
    x2 = coord_2[:,1]
    
    [YC, YR] = np.meshgrid(y2, y1)
    [XC, XR] = np.meshgrid(x2, x1)

    dR = np.sqrt( (YR-YC)**2 + (XR-XC)**2)
    
    return dR
    

