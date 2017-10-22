'''
New trial of drift correction
'''
import pyfftw
import numpy as np
import src.shared_funcs.tifffunc as tf



def pyfftw_container(ny, nx, bwd = False):
    '''
    construct a fftw container to perform fftw.
    '''
    a = pyfftw.empty_aligned((ny,nx),dtype = 'complex128')
    b = pyfftw.empty_aligned((ny,nx),dtype = 'complex128')
    if bwd:
        container = pyfftw.FFTW(a,b,axes = (0,1))
    else:
        container = pyfftw.FFTW(a,b,axes = (0,1),direction = 'FFTW_BACKWARD')
    return container


def cross_corr_phase_spectrum(stack_1, stack_2):
    nz, ny, nx = stack_1.shape
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)
    for ii in range(nz):
        container_1(stack_1[ii])
        container_2(stack_2[ii])
        ft_1 = container_1.get_output_array()
        ft_2 = container_2.get_output_array()
        F_prod = np.conj(ft_1)*ft_2
        phase_spec = F_prod/np.absolute(F_prod)
        container_invx(phase_spec) # instead of using the 
        corr_spec = np.abs(container_invx.get_ouput_array())




