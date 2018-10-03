'''
Additional signal munging that can be shared among analysis pipelines
peak finding, flag_decode, matrix interleaving, etc.
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
import pyfftw
import scipy.fftpack as fftpack
from scipy import signal
import src.analysis.clustering as clustering
from scipy.stats import norm


def interleave(mat1, mat2, direction = 'c'):
    '''
    interleave two matrices either by row or by columns. default by column.
    '''
    Y1, X1 = mat1.shape
    Y2, X2 = mat2.shape
    if (Y1!=Y2 or X1!=X2):
        print("The matrix dimensions do not match.")
        return

    if direction == 'c':
        cc = np.c_[mat1.ravel(), mat2.ravel()]
        inter_mat = cc.reshape((Y1, 2*X1))
    else:
        cc = np.c_[mat1.T.ravel(), mat2.T.ravel()]
        inter_mat = cc.reshape((Y1, 2*X1)).T

    return inter_mat



def flag_decode(n_flag, n_mask = 294):
    if n_flag < 294:
        return n_flag
    else:
        flags = []
        pw = int(np.log(float(n_flag))/np.log(294.0)) + 1 # log(294, n_flag)
        n_remain = n_flag
        for nw in range(pw):
            root = int(np.power(294, pw-nw))
            qt, rm = np.divmod(n_remain, root)
            flags.append(int(qt))
            n_remain = rm

        flags.append(rm)

        return flags

def freq_cut(dff_raw, fcut, dt = 0.5):
    '''
    dff_raw: the raw Delta F/F signals, each column represents DF/F of a cell.
    fcut: the cutting frequency
    dt: time step
    return: the indices of the cells that show little activities.
    '''
    ft_dff = fftpack.fftshift(fftpack.fft(dff_raw, axis = 0)) # zero frequency at center 
    NT, NP = dff_raw.shape
    T = NT*dt # the total duration
    fNyq = 0.5/dt # Nyquist frequency
    df= 2*fNyq/NT # the frequency resolution
    Ncut = int(np.min([fNyq,fcut])/df) # where to cut off

    return ft_dff, Ncut



def coord_edgeclean(coord_3d, edge_pos, dim = 'x', direct = 1):
    '''
    Remove the fake blobs within crop_width around edge pixels, i.e., if the blob center is beyond the edge px by crop_width pixels, discard that blob.
    direct is positive: the discarded blobs should have center position larger than the pixel
    compiled_data should have the coord key and the 'data' key.
    '''
    if dim == 'x':
        c_list = coord_3d[:,2] # take out the last column
    elif dim == 'y':
        c_list = coord_3d[:,1] # take out the second column

    if direct ==1:
        ind_discard = np.where(c_list > edge_pos)[0]
    else:
        ind_discard = np.where(c_list < edge_pos)[0]
    return ind_discard



def stimuli_trigger_period(T, dt, NT, hl_ratio, t_off):
    '''
    T: period
    dt: time steps
    NT: Number of time points
    hl_ratio: duty, the fraction of high-level in each period
    t_off: offset, must between 0 and T
    '''
    sig_raw = signal.square(2*np.pi*np.arange(NT)*dt/T, duty = hl)
    if(t_off ==0):
        return sig_raw
    else:
        n_shift = int(np.round(t_off/dt))
        sig_shift  = np.roll(sig_raw, n_shift)
        return sig_shift

def stimuli_trigger_arbitrary(dt, NT, t_sti, d_sti, t_shift = 0., mode = 'q'):
    '''
    dt: time step
    NT: number of time points
    t_sti: stimulation onset time
    d_sti: duration of stimulation
    t_shift: how much to shift back the onset. Default: 0
    a_sti: amplitude of stimulation, default 1.0
    '''
    sig_sti = np.zeros(NT)
    t_duration = np.ceil(d_sti/dt).astype('int')
    print(t_duration)
    t_back = np.ceil(t_shift/dt).astype('int')
    t_onset = (t_sti//dt - t_back).astype('int')
    t_onset[t_onset < 0] = 0

    if mode == 'q':
        # square wave
        for nt in t_onset:
            sig_sti[nt:nt+t_duration]=1.
    elif mode == 'e':
        # exponential decay
        a = 1./(1.-np.exp(-3*t_duration*dt))
        b = 1.-a
        for nt in t_onset:
            sig_sti[nt:nt+3*t_duration]=a*np.exp(-np.arange(3*t_duration)*dt)+b

    return sig_sti


