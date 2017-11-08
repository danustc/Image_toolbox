'''
Removing high-frequency noises (those wrongly extracted cells)
Remove the high-frequency noise on the edgeds
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
import pyfftw
import scipy.fftpack as fftpack
import src.networks.clustering as clustering

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
        c_list = coord_3d[:,-1] # take out the last column
    elif dim == 'y':
        c_list = coord_3d[:,1] # take out the second column

    if direct ==1:
        ind_discard = np.where(c_list > edge_pos)[0]
    else:
        ind_discard = np.where(c_list < edge_pos)[0]
    return ind_discard


