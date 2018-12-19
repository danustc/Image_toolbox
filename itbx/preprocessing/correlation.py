'''
The core algorithms of cross correlation
'''
import pyfftw
import numpy as np
from scipy import signal



def _phase_construct_(row_range, col_range, n_denom, forward = True, shift_r = False, shift_c = False ):
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

    if shift_r:
        rvec = np.fft.fftshift(rvec)
    if shift_c:
        cvec = np.fft.fftshift(cvec)

    ex_in = 2*np.pi*np.outer(rvec,cvec)/n_denom
    re_exin = np.cos(ex_in)
    im_exin = np.sin(ex_in)
    if forward:
        return re_exin - im_exin*1j
    else:
        return re_exin + im_exin*1j


def fft_image(img, abs_only = True):
    '''
    return the fourier transform of the image with the zero-frequency component shifted to the center.
    '''
    NY, NX = img.shape
    ct = pyfftw_container(NY, NX)
    ct(img)
    ft_img = ct.get_output_array()
    ft_img = np.fft.fftshift(ft_img) # move the zero-freq components to the center
    if abs_only:
        return np.abs(ft_img)
    else:
        return ft_img

def ift_image(imf):
    '''
    inversely Fourier transform the images
    '''
    NY,NX = imf.shape
    ct = pyfftw_container(NY, NX, bwd = True)
    imf = np.fft.fftshift(imf) # shift the imf first
    ct(imf)
    img = ct.get_output_array()

    return np.abs(img)



def bpf_image(frame, p_low, p_high):
    '''
    create a bandpass filter
    '''
    ft_img = fft_image(frame, abs_only = False)
    NY, NX = frame.shape
    YL, YH = int(NY*p_low), int(NY*p_high)
    XL, XH = int(NX*p_low), int(NX*p_high)
    



def pyfftw_container(ny, nx, bwd = False):
    '''
    construct a fftw container to perform fftw.
    '''
    a = pyfftw.empty_aligned((ny,nx),dtype = 'complex128')
    b = pyfftw.empty_aligned((ny,nx),dtype = 'complex128')
    if bwd:
        container = pyfftw.FFTW(a,b,axes = (0,1),direction = 'FFTW_BACKWARD')
    else:
        container = pyfftw.FFTW(a,b,axes = (0,1),direction = 'FFTW_FORWARD')
    return container


def high_pass(img_k, k_frac = 0.01):
    '''
    Assumption: the r-resolution is the same in x and y direction.
    remove the low frequency components around
    img_k: the Fourier transformed image in k-space, the 0-component at the corner (not shifted)
    '''
    KY, KX = img_k.shape
    hky, hkx = int(KY//2), int(KX//2)
    kspec_y, kspec_x = np.arange(KY) - hky, np.arange(KX) - hkx
    [MKX, MKY] = np.meshgrid(kspec_x/hkx, kspec_y/hky)
    hp_indicator = MKX**2 + MKY**2 > (k_frac**2)
    valid_index = np.where(np.fft.fftshift(hp_indicator))
    return valid_index


def cross_corr_shift_frame(im_1, im_2, container_1 = None, container_2 = None, container_inv = None, hf = True, hanning_2d = None, up_rate= None):
    '''
    calculate cross-correlation based shift between two images
    assumption: the two images are in same size.
    precision: to single pixels
    '''
    N, M  = im_1.shape
    hy = int(N//2)
    hx = int(M//2)
    if container_1 is None:
        container_1 = pyfftw_container(N,M)
    if container_2 is None:
        container_2 = pyfftw_container(N,M)
    if container_inv is None:
        container_inv = pyfftw_container(N,M, bwd = True)

    if hf: # apply hanning filter to the images
        if hanning_2d is None:
            hann_w = signal.hann(M)
            hann_h = signal.hann(N)
            hanning_2d = np.outer(hann_h, hann_w)

        container_1(im_1*hanning_2d)
        container_2(im_2*hanning_2d)
    else: # not applying the hanning filter
        container_1(im_1)
        container_2(im_2)

    ft_1 = container_1.get_output_array()
    ft_2 = container_2.get_output_array()

    F_prod = np.conj(ft_1)*ft_2
    phase_spec = F_prod/np.absolute(F_prod) # phase_spec
    container_inv(F_prod) # invert phase spec or cross-correlation function?
    corr_spec = np.fft.fftshift(np.abs(container_inv.get_output_array()))
    shy, shx = np.unravel_index(np.argmax(corr_spec), (N,M)) # find the maximum index of normalized correlation matrix

    #if shy > hy:
    #    shy = -(N-shy)
    #if shx > hx:
    #    shx = -(M-shx)
    shy -=hy
    shx -=hx
    if up_rate is None:
        return shy, shx # this can be used for the initial guess, the Fourier transform product is also returned
    else:
        # upsampling
        phase_N = _phase_construct_([(shy-2)*up_rate, (shy+2)*up_rate+1], [-hy, N-hy], N*up_rate, forward=False, shift_c = True)
        phase_M = _phase_construct_([-hx, M-hx], [(shx-2)*up_rate, (shx+2)*up_rate+1], M*up_rate, forward=False, shift_r = True)

        corr_spec_nbhd = np.abs(np.matmul(np.matmul(phase_N, phase_spec), phase_M)) # cross correlation in the neighborhood of (x0,y0)
        iny, inx = np.unravel_index(np.argmax(corr_spec_nbhd), corr_spec_nbhd.shape)

        shy = (iny-2*up_rate)/up_rate + shy
        shx = (inx-2*up_rate)/up_rate + shx
        return shy, shx, corr_spec_nbhd
        # OK this passes test.



def cross_corr_stack_self(stack, adj_ref = False, verbose = True, pivot_slice = 0):
    '''
    Align a stack to itself. If adj_ref is True, then align each slice to the neighboring slice preceeding it.
    Added: subpixel precision
    '''
    nz, ny, nx = stack.shape
    hy = ny//2
    hx = nx//2
    shift_coord = np.zeros([nz, 2])
    hann_w = signal.hann(nx)
    hann_h = signal.hann(ny)
    hfilter = np.outer(hann_h, hann_w)

    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)
    if np.isscalar(pivot_slice):
        ref_frame = stack[pivot_slice]
    else:
        ref_frame = pivot_slice

    for ii in range(nz):
        shy, shx  = cross_corr_shift_frame(ref_frame, stack[ii], container_1, container_2, container_invx, hf = True, hanning_2d = hfilter)[:2]
        if verbose:
            print("slice ", ii+1, '-->', shy, shx)
        shift_coord[ii] = np.array([-shy, -shx])
        # then we have to shift im 2 by (-shy, -shx)
        if adj_ref:
            # if the stack is aligned in the adjacent mode, them each slice should be updated in place; otherwise we can just return the shift coordinates.
            shifted_frame = interpolation.shift(stack[ii+1],shift = [-shy, -shx])
            ref_frame = shifted_frame
            stack[ii+1] = shifted_frame

    return shift_coord # Hmmmm, this is much nicer.

