"""
This is df/f calculation based on the paper: Nature protocols, 6, 28–35, 2011
Created by Dan on 08/18/16
Last update: 05/09/18
"""

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
package_path =r"C:\Users/Admin/Documents/GitHub/Image_toolbox\\"
sys.path.append(package_path)
import numpy as np
from src.shared_funcs.numeric_funcs import smooth_lpf, gaussian2d_fit, gaussian1d_fit
from scipy.signal import exponential, fftconvolve
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
def adjacent_2d_hist(t_trace):
    '''
    create a 2-D histogram for adjacent pairs of data points
    '''

    Zn = t_trace[:-1]
    Zp = t_trace[1:]
    Z_diff = (Zp-Zn)/np.sqrt(2.)
    Z_sum  = (Zp+Zn)/np.sqrt(2.)

    return Z_diff, Z_sum



def background_suppress(signal, ind_A, ext = 2):
    '''
    set background points to 0.
    ext: number of pixels extended from the root of each peak
    '''
    signal_bgsup = np.copy(signal)
    nsig = signal.size
    u1 = np.union1d(ind_A, ind_A+1)
    u2 = np.union1d(u1, ind_A+2)
    d1 = np.union1d(u2, ind_A-1)
    d2 = np.union1d(d1, ind_A-2)
    peak_ind = np.where(d2 >=0)[0] # do not take negative indices
    mask = np.ones(nsig, dtype = bool)
    mask[peak_ind] = False # mask out the peak indices
    signal_bgsup[mask] = 0.
    return signal_bgsup


def min_window(shit_data, wd_width):
    """
    Calculate the baseline
    """
    f0 = np.zeros_like(shit_data)
    f0[:wd_width] = np.amin(shit_data[:wd_width], axis = 0)

    N = len(f0)
    for ii in np.arange(wd_width, N-wd_width):
        f0[ii] = np.amin(shit_data[ii-wd_width:ii+wd_width], axis = 0)

    f0[N-wd_width:] = np.amin(shit_data[N-wd_width:], axis = 0)

    return f0
# this is a good baseline calculation.


def dff_raw(shit_data, ft_width, ntruncate = 20):
    """
    calculate df_f for shit_sig.
    ft_width: the time scale of features in the raw f(t). Unit: 1 (not in seconds)
    ntruncate: the number of datapoits to be discarded.
    Get both F0 and df_f. Correct the baseline if necessary.
    """
    shit_data += 1.0e-06
    s_filt = smooth_lpf(shit_data[ntruncate:], ft_width)[1]
    f_base = min_window(s_filt, 6*ft_width) + 2.0e-08
    dff_r = (shit_data[ntruncate:]-f_base)/f_base


    return dff_r


def dff_hist(dff_r, nbin = 200, range = None):
    '''
    assumption: dff is already calculated.
    This is for 1-D array dff only.
    After hilltop, there is no need to fit.
    '''
    #m,s = stats.norm.fit(dff_r) does not work. More constraints should be added to the fit.
    hist, be = np.histogram(dff_base, nbin, range = (rbottom, rtop)) # calculate the histogram of baseline


    try:
        popt, pcov  = gaussian1d_fit(xx,hist,x0 = mu,sig_x = 0.05, A = A0, offset = 0.)
        s = np.sqrt(0.5/popt[1])
        #print("noise level:", s)

        return hist, s  # dff_zscore = dff_r/s
    except RuntimeError:
        print("Fitting failed.")
        return hist, -1

def dff_AB(dff_r, gam = 0.05, nbins = 40):
    '''
    Using Bayesian theory to infer the identity of the data points (signal or noise)
    update on 07/24/2018: reset the prior.
    basecut: set the bottom fraction of datasets as baseline.
    '''
    ND = len(dff_r)

    Z_diff, Z_sum = adjacent_2d_hist(dff_r)

    sum_range, m_sum, s_sum = hillcrop_base_finding(Z_sum , niter = 25, conf_level = 1.5)
    m_diff = np.mean(Z_diff)
    s_diff = np.std(Z_diff)

    diff_range = np.logical_and((Z_diff - m_diff) < 0.8*s_diff, (Z_diff - m_diff) > -0.5*s_diff) # This is way too random
    #diff_range = np.logical_and((Z_diff - av_diff) < s_diff, (Z_diff - av_diff) > s_diff) # This is way too random
    B_indices = np.logical_and(diff_range, sum_range)
    B_diff = Z_diff[B_indices]
    B_sum = Z_sum[B_indices]
    md,sd = stats.norm.fit(B_diff)
    ms,ss = stats.norm.fit(B_sum) # ms is the recognized noise level


    dmin, dmax, smin, smax = Z_diff.min(), Z_diff.max(), Z_sum.min(), Z_sum.max()
    del_diff, del_sum = (dmax-dmin)/(nbins-1), (smax-smin)/(nbins-1)
    h2, ne, pe = np.histogram2d(Z_diff, Z_sum, bins = nbins, range = [[dmin-del_diff*0.5, dmax+del_diff*0.5],[smin-del_sum*0.5, smax+del_sum*0.5]]) #2d histogram distribution of (Zn_k, Zn_k+1)
    rv = stats.multivariate_normal(mean = [md, ms], cov = [[sd**2, 0.], [0., ss**2]]) # normal distribution for zero-correlation
    PZB = rv.pdf(np.dstack((Z_diff,Z_sum))) # the distribution of B: dual-variate Gaussian


    #h1, _, _ = np.histogram2d(B_diff, B_sum, bins = nbins, range = [[dmin-del_diff*0.5, dmax+del_diff*0.5],[smin-del_sum*0.5, smax+del_sum*0.5]])
    #B_dist = h1/h1.sum()


    Z_dist = h2/h2.sum() # normalized distribution
    ind_zn = np.searchsorted(ne, Z_diff)-1 #indices of %Zn in the histogram 
    ind_zp = np.searchsorted(pe, Z_sum)-1 #indices of Zn+1 in the histogram
    PZ = Z_dist[ind_zn, ind_zp] # the probability of data transitions
    #PZB = B_dist[ind_zn, ind_zp] # a test added by Dan on 10/17/18
    PBZ = PZB/(PZ+0.001) # Bayesian theory, the activity posterior probability 
    beta = gam/(1.+gam)
    id_A = np.where(np.logical_and(PBZ<beta, Z_sum>m_sum-0.2*s_sum))[0] # add more constraint: the sum must be larger than -1 std
    #whether_A = np.logical_and(PBZ<beta, (Z_sum-ms)>-ss) # add more constraint: the sum must be larger than -1 std
    id_peak = np.union1d(id_A, id_A+1)
    peak_mask = np.ones(ND, dtype=bool)
    peak_mask[id_peak] = False
    bg_points = dff_r[peak_mask]
    background = np.mean(bg_points)
    #id_peak = np.intersect1d(id_peak, np.where(dff_r > background_test)[0])
    #bg_points = dff_r[id_peak]

    noise_level = np.std(bg_points)
    return id_peak, background, noise_level


def activity_level(dff_trace, baseline = 0, tt = None, upsampling = 2, background_sup = False):
    '''
    calculate the activity level of a time trace.
    '''
    if background_sup:
        # background suppression
        dff_trace[dff_trace < baseline] = 0.
    if upsampling ==1:
        al = (dff_trace-baseline).sum()
    else:
        NT = len(dff_trace)
        if tt is None:
            tt = np.arange(NT)
        tmid = np.arange(upsampling*(NT-1)+1)/upsampling
        f = interp1d(tt, dff_trace)
        dff_interp = f(tmid)
        al = (dff_interp-baseline).sum()/upsampling

    return al



def dff_expfilt(dff_r, dt, t_width = 2.0, savefilter = False):
    """
    Exponentially weighted moving average filter
    OK this also works.
    """
    M = int(t_width/dt+1)*8 + 1 # the number of window
    wd = exponential(M, center=None, tau = t_width) # Symmetric = True
    NT = len(dff_r)
    tt = np.arange(1,NT+1)*dt
    denom_filter = (1-np.exp(-tt/t_width))*t_width # the denominator
    dff_expf = fftconvolve(dff_r, wd, mode='same')*dt/denom_filter
    if savefilter:
        return dff_expf, wd
    else:
        return dff_expf

def dff_expfilt_group(dff_r, dt, t_width = 2.0):
    """
    Exponentially weighted moving average filter
    OK this also works.
    Test test
    """
    dft = dff_r.T
    dff_expf = np.array([dff_expfilt(fr_col, dt, t_width) for fr_col in dft]).T
    #dff_expf = np.zeros([NT, NP])
    print(dff_expf.shape)

    return dff_expf

def hillcrop_base_finding(dff_r, niter = 4, conf_level = 3):
    '''
    find pulses by cropping hill tops
    '''
    dff_c = np.copy(dff_r)
    for ii in range(niter):
        m_dff = np.mean(dff_c)
        s_dff = np.std(dff_c)
        base_ind = (dff_r -m_dff) < (conf_level*s_dff)
        dff_c = dff_r[base_ind]

    return base_ind, m_dff, s_dff


