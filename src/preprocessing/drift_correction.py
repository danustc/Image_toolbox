'''
New trial of drift correction
'''
import pyfftw
import numpy as np


def _phase_construct_(row_range, col_range, n_denom, forward = True):
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
        return re_exin - im_exin*1j
    else:
        return re_exin + im_exin*1j

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


def cross_corr_shift_frame(im_1, im_2, container_1 = None, container_2 = None, container_inv = None):
    '''
    calculate cross-correlation based shift between two images
    assumption: the two images are in same size.
    precise to single pixels
    '''
    N, M  = im_1.shape
    if container_1 is None:
        container_1 = pyfftw_container(N,M)
    if container_2 is None:
        container_2 = pyfftw_container(N,M)
    if container_inv is None:
        container_inv = pyfftw_container(N,M, bwd = True)

    container_1(im_1)
    container_2(im_2)
    ft_1 = container_1.get_output_array()
    ft_2 = container_2.get_output_array()

    F_prod = np.conj(ft_1)*ft_2
    phase_spec = F_prod/np.absolute(F_prod) # phase_spec

    container_inv(phase_spec)
    corr_spec = np.abs(container_inv.get_output_array())
    y0, x0 = np.unravel_index(np.argmax(corr_spec), (N,M)) # find the maximum index of normalized correlation matrix

    return y0, x0, F_prod # this can be used for the initial guess, the Fourier transform product is also returned

def cross_corr_shift_stack(stack_1, stack_2=None, estm_dft = False, init_guess = None, up_rate = 2):
    '''
    calculate the cross correlation
    Added: subpixel precision
    '''
    nz, ny, nx = stack_1.shape
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)

    if stack_2 is None: # align the stack to itself, every frame to its adjacent frame
        for ii in range(nz-1):
            if estm_dft:
                y0, x0, F_prod = cross_corr_shift_frame(stack_1[ii], stack_1[ii+1], container_1, container_2, container_invx)
            else: # no estimation of the shift
                container_1(stack_1[ii])
                container_2(stack_1[ii+1])
                F_prod = np.conj(container_1.get_output_array())*container_2.get_output_array()

                if init_guess is None:
                    y0 = 0
                    x0 = 0
                else:
                    y0, x0 = init_guess

            # construct the phase matrix with smaller dimensions
            phase_N = _phase_construct_([-2*up_rate+y0, up_rate*2+1+y0], ny, ny*up_rate, forward=False)
            phase_M = _phase_construct_(nx, [-2*up_rate+x0, up_rate*2+1+x0], nx*up_rate, forward=False)

            coor_spec_nbhd = np.matmul(np.matmul(phase_N, F_prod), phase_M) # cross correlation in the neighborhood of (x0,y0)
            iny, inx = np.unravel_index(np.argmax(corr_spec_nbhd), (up_rate*2, up_rate*2))

            shy = (iny-y0)/up_rate
            shx = (inx-x0)/up_rate # the real shift in original pixel

        return shy, shx

    else: # align one stack to the other
        for ii in range(nz):
            if estm_dft:
                y0, x0, F_prod = cross_corr_shift_frame(stack_1[ii], stack_2[ii], container_1, container_2, container_invx)
            else:
                container_1(stack_1[ii])
                container_2(stack_2[ii])

                F_prod = np.conj(container_1.get_output_array())*container_2.get_output_array()

                if init_guess is None:
                    y0 = 0
                    x0 = 0
                else:
                    y0, x0 = init_guess

            # construct the phase matrix with smaller dimensions
            phase_N = _phase_construct_([-2*up_rate+y0, up_rate*2+1+y0], ny, ny*up_rate, forward=False)
            phase_M = _phase_construct_(nx, [-2*up_rate+x0, up_rate*2+1+x0], nx*up_rate, forward=False)

            coor_spec_nbhd = np.matmul(np.matmul(phase_N, F_prod), phase_M) # cross correlation in the neighborhood of (x0,y0)
            iny, inx = np.unravel_index(np.argmax(corr_spec_nbhd), (up_rate*2, up_rate*2))

            shy = (iny-y0)/up_rate
            shx = (inx-x0)/up_rate # the real shift in original pixel

        return shy, shx
