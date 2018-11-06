'''
Created by Dan on 08/15/16
This file contains several small functions shared among all the classes.
This one has Numerical functions. For graphic functions, see graph_funcs.py
Adapted from Scipy cookbook.
Last update: 11/06/2018
'''

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import optimize
from scipy.ndimage import gaussian_filter1d


def marcenko_pastur(q,sig = 1.,nl = 100):
    '''
    Marcenko-pastur distribution
    '''
    sig2 = sig*sig
    isq = 1./np.sqrt(q)
    l_min, l_max = sig2*(1-isq)**2, sig2*(1+isq)**2
    lamb = np.linspace(l_min, l_max, nl)
    p_lam = q/(2*np.pi*sig2)*np.sqrt((l_max-lamb)*(lamb-l_min))/lamb
    return p_lam, lamb



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


def gaussian1D(xx, x0, aa, A, ofst= 0.):
    '''
    sx^2 = sqrt(0.5/aa)
    '''
    x0 = float(x0)
    k = aa*(xx-x0)**2
    g = A*np.exp(-k) + ofst
    return g.ravel()

def gaussian2D(ds_xy, x0, y0,a,b,c, A, ofst=0.):
    '''
    rho = c/(2*sqrt(ab))
    sx^2 = 2b/(4ab-c)
    sy^2 = 2a/(4ab-c)
    '''
    x0 = float(x0)
    y0 = float(y0)
    x = ds_xy[:,:,0]
    y = ds_xy[:,:,1]
    #g = A*np.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2+(x-x0)*(y-y0)/(ss*ss))/2)+ofst
    k = a*(x-x0)**2 + b*(y-y0)**2 - c*(x-x0)*(y-y0)
    g = A*np.exp(-k) + ofst

    return g.ravel()


def gaussian1d_fit(xx, data, x0, sig_x, A, offset):
    '''
    fit to a 1d gaussian function with initial guess
    '''
    a0 = 0.5/(sig_x**2)
    initial_guess = (x0, a0, A, offset)
    popt, pcov = optimize.curve_fit(gaussian1D, xx, data, p0 = initial_guess)
    return popt, pcov


def gaussian2d_fit(x,y, data, x0, y0, sig_x, sig_y, rho, A, offset):
    '''
    fit to a 2d gaussian function with initial guess
    rho: correlation, between[-1,1]
    '''
    a0 = 0.5/(np.sqrt(1-rho**2)*sig_x**2)
    b0 = a0*sig_x**2/(sig_y**2)
    c0 = a0*2*rho*sig_x/sig_y
    initial_guess = (x0, y0, a0,b0,c0, A, offset)
    popt, pcov = optimize.curve_fit(gaussian2D, np.dstack((x,y)), data, p0=initial_guess)
    return popt, pcov


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
    To find the mask of a small patch on a big frame.
    frame_size: ny, nx
    cr: center of the blob, unit in pixel
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
        # within bound check
        msel = np.logical_and(msel_1, msel_2)
        ind_mask = (my[msel],mx[msel])
        return ind_mask

    # end of circ_mask

def circ_mask_patch_group(frame_size, crs, dr):
    '''
    return a list of patch indices
    '''
    ind_mask_y = None
    ind_mask_x = None

    return ind_mask_y, ind_mask_x



def spheri_mask_patch(frame_size, cr, dr, pxl):
    '''
    frame_size:(nz, ny, nx)
    cr:(cz, cy, cx), unit micron
    dr:The radius of the blob, unit micron
    pxl: pixel size (pz, py, px)
    '''
    sz, sy, sx = frame_size
    # convert to unit of pixels
    idx_upper = ((cr+dr)//pxl+1).astype('int')
    idx_lower = ((cr-dr)//pxl).astype('int')
    n_begin = int(dr)+1
    zz = np.arange(idx_lower[0], idx_upper[0])*pxl[0]-cr[0]
    yy = np.arange(idx_lower[1], idx_upper[1])*pxl[1]-cr[1]
    xx = np.arange(idx_lower[2], idx_upper[2])*pxl[2]-cr[2]

    [MY, MZ, MX] = np.meshgrid(yy,zz,xx)
    RR = np.sqrt( MY**2 + MZ**2 + MX**2)
    idz, idy, idx = np.where(RR<dr)
    inr = np.c_[idz, idy, idz] + idx_lower
    return inr



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

def solid_angle(vec):
    '''
    calculate the solid angle of a vector (theta, phi)
    vec can be a list
    '''
    print(vec.shape)
    nn = np.linalg.norm(vec,axis = 1)
    nvec = (vec.T/nn).T
    theta = np.arccos(nvec[:,2]) #theta
    phi = np.angle(nvec[:,0]+1j* nvec[:,1])
    return theta, phi
