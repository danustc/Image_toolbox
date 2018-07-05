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
from scipy import stats
import matplotlib.pyplot as plt
import pyfftw

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

    hist, be =np.histogram(dff_r, bins = 100) # histogram
    hind = np.argmax(hist)
    mu = (be[hind]+be[hind+1])*0.5 # the average of mu
    if mu>0:
        fb_new = f_base*(1.+mu)
        dff_r = (shit_data[ntruncate:]-fb_new)/fb_new

    return dff_r
    # done with dff_raw


def dff_hist(dff_r, nbin = 100, rect= False, noise_norm = False):
    '''
    assumption: dff is already calculated.
    '''
    #m,s = stats.norm.fit(dff_r) does not work. More constraints should be added to the fit.
    hist, be = np.histogram(dff_r, nbin) # fit with histogram
    mu_ind = np.argmax(hist) # find the maximum
    A0 = hist[mu_ind] # the initial guess of the coefficient
    print(A0)

    xx = (be[:-1]+be[1:])*0.50
    popt, pcov  = gaussian1d_fit(xx,hist,x0 = 0.0,sig_x = 0.10, A = A0/2, offset = 0.)
    s = np.sqrt(0.5/popt[1])
    print("noise level:", s)

    dff_n = dff_r
    if rect:
        dff_n[dff_r<0] = 0. #rectify so that there are no negative dff values.
    if noise_norm: # normalize to the noise level
        dff_n/=s # this is actually signal-to-noise level

    return dff_n,  popt # dff_zscore = dff_r/s

def dff_AB(dff_r, gam = 0.05):
    '''
    Using Bayesian theory to infer the identity of the data points (signal or noise)
    '''
    Zn = dff_r[:-1]
    Zp = dff_r[1:]
    h2, ne, pe = np.histogram2d(Zn, Zp, bins = 50) #2d histogram distribution of (Zn_k, Zn_k+1)
    hne = (ne[:-1]+ne[1:])/2.
    hpe = (pe[:-1]+pe[1:])/2.
    [NE, PE] = np.meshgrid(hne, hpe)
    popt, pcov = gaussian2d_fit(x = NE, y = PE, data = h2.ravel(), x0=0., y0=0., sig_x = 0.5, sig_y = 0.5, rho = 0.60, A = 10, offset = 0.10)
    mx, my, a,b,c, A, ofst = popt # the fitted parameters
    sxq = 2*b/(4*a*b-c*c)
    syq = 2*a/(4*a*b-c*c)
    sxy = np.sqrt(sxq*syq)
    rho = c/(2.*np.sqrt(a*b))
    rv = stats.multivariate_normal(mean = [mx, my], cov = [[sxq, rho*sxy], [rho*sxy, syq]])
    PZB = rv.pdf(np.dstack((Zn,Zp))) # the distribution of B: dual-variate Gaussian
    Z_dist = h2/h2.sum() # normalized distribution
    ind_zn = np.searchsorted(ne, Zn)-1 #indices of %Zn in the histogram 
    ind_zp = np.searchsorted(pe, Zp)-1 #indices of Zn+1 in the histogram
    PZ = Z_dist[ind_zn, ind_zp] # the probability of data transitions
    PBZ = PZB/(PZ+0.001) # Bayesian theory, the activity posterior probability 
    beta = gam/(1.+gam)
    id_A = np.where(PBZ<beta)[0]
    return id_A




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

#-------------------------Frequency domain----------------------------
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

