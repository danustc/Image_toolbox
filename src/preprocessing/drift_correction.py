'''
New trial of drift correction
'''
import pyfftw
import numpy as np


def _phase_construct(row_range, col_range, n_denom, forward = True):
    '''
    row_range, col_range: the pixel range in rows and cols. If has two elements, then taken as the lower and upper bound; otherwise, taken as range(N).
    n_denom: Denominator N.
    forward: if true, -2pi*i; otherwise 2pi*i.
    '''
    if np.isscalar(row_range):
        rvec = np.arange(row_range)
    else:
        rvec = np.arange(*row_range)

    if np.isscalar(col_range):
        cvec = np.arange(col_range)
    else:
        cvec = np.arange(*col_range)

    ex_in = 2*np.pi*np.outer(rvec,cvec)/n_denom
    re_exin = np.cos(ex_in)
    im_exin = np.sin(ex_in)
    if forward:
        return re_exin - j*im_exin
    else:
        return re_exin + j*im_exin

def _corr_matmul_(F,G, inv = False):
    '''
    compute 2D discrete Fourier transform by matrix multiplication.
    '''
    N,M = F.shape
    coef_N = np.linspace(N)/N
    coef_M = np.linspace(M)/M
    ex_N = np.cos(2*np.pi*coef_N) + j*np.sin(2*np.pi*coef_N)
    ex_M = np.cos(2*np.pi*coef_M) + j*np.sin(2*np.pi*coef_M)


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


def cross_corr_phase_spectrum(stack_1, stack_2=None, estm_dft = False, init_guess = None, up_rate = 2):
    '''
    calculate the cross correlation
    Added: subpixel precision
    '''
    nz, ny, nx = stack_1.shape
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)
    if stack_2 is None:
        for ii in range(nz-1):
            container_1(stack_1[ii])
            container_2(stack_1[ii+1])
            ft_1 = container_1.get_output_array()
            ft_2 = container_2.get_output_array()
            F_prod = np.conj(ft_1)*ft_2
            phase_spec = F_prod/np.absolute(F_prod) # phase_spec
            if estm_dft:
                container_invx(phase_spec) # instead of using the direct xcross, we use the phase spec, which may increase the precision. 
                corr_spec = np.abs(container_invx.get_output_array())
                iny, inx = np.unravel_index(np.argmax(corr_spec), (ny, nx)) # find the maximum index of normalized correlation matrix
            else: # no estimation of the shift
                if init_guess is None:
                    y0 = 0.
                    x0 = 0.
                else:
                    y0, x0 = init_guess

            # construct the phase matrix with smaller dimensions
            phase_N = _phase_mat(1.5*up_rate, ny, y0, up_rate)
            phase_M = _phase_mat(1.5*up_rate, nx, x0, up_rate)






    else:
        for ii in range(nz):
            container_1(stack_1[ii])
            container_2(stack_2[ii])
            ft_1 = container_1.get_output_array()
            ft_2 = container_2.get_output_array()
            F_prod = np.conj(ft_1)*ft_2
            phase_spec = F_prod/np.absolute(F_prod) # phase_spec
            container_invx(phase_spec) # instead of using the direct xcross, we use the phase spec, which may increase the precision. 
            corr_spec = np.abs(container_invx.get_output_array())
            iny, inx = np.unravel_index(np.argmax(corr_spec), (ny, nx)) # find the maximum index of normalized correlation matrix
            print('Max ind:', iny, inx)
            stack_2[ii] = np.roll(stack_2[ii], -iny, axis = 0)
            stack_2[ii] = np.roll(stack_2[ii], -inx, axis = 1)
        return stack_2

