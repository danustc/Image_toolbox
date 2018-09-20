'''
Analysis in frequency domain
'''
import numpy as np
import pyfftw

def spectrogram(dff_r, dt, twindow, kt = None, k_frac = 1.):
    '''
    create spectrogram of the dff_r signal (single)
    dt: time step in the original data file
    twindow: the width of the time window
    kt: the time step of the spectral gram, if None, set to half of the time window
    '''
    NT = dff_r.size
    NK = int(twindow // dt)
    if kt is None:
        jump_kt = int(NK//2) # convert to number of time steps
    else:
        jump_kt = int(kt/dt) # convert to number of time steps

    NW = int((NT-NK)/jump_kt)+1
    sg_index = (np.tile(np.arange(NK),[NW, 1]).T + np.arange(NW)*jump_kt).T
    dff_km = dff_r[sg_index]
    a = pyfftw.empty_aligned((NW, NK), dtype = 'complex128')
    b = pyfftw.empty_aligned((NW, NK), dtype = 'complex128')
    fft_ax0 = pyfftw.FFTW(a,b) # Only do Fourier Transforn across the last dimension
    fft_ax0(dff_km)
    spec_comp = fft_ax0.get_output_array()
    HK = int(NK*k_frac/2)
    sgram  = np.abs(spec_comp[:,:HK])
    kmax  = HK/(dt*NK) # the maximum of k
    return sgram.T, kmax # horizontal: time; vertical: frequency


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


def spectral_power(sg_gram, k_max, kr, ex_zero = True):
    '''
    calculate the spectral power within a range.
    sg_gram: spectrogram, NK x NW matrix, amplitudes of the Fourier components
    k_max: the maximum frequency of the sg_gram
    kr_ the range (unitless of frequency to be integrated)
    n_begin: if 0, include the 0-frequency component, otherwise ex
    '''
    NK, NW = sg_gram.shape
    dk = k_max / NK # the k_step
    if np.isscalar(kr):
        k_up = int(np.ceil(kr*NK))
        k_down = int(ex_zero)
    else:
        k_down = int(kr[0]*NK)
        k_up = int(np.ceil(kr[1]*NK))

    spower = np.sum(sg_gram[k_down:k_up]**2, axis = 0)
    return spower



def main():
    pass

if __name__ == '__main__':
    main()

