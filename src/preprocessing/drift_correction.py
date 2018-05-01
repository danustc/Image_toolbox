'''
New trial of drift correction
'''
import pyfftw
import numpy as np
from scipy.ndimage import interpolation
from tifffile import TiffFile

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


def shift_stack(stack, shift_coord):
    '''
    shift a stack via interpolation
    '''
    nz, _, _ = stack.shape
    for iz in range(1,nz):
        shifted_frame = interpolation.shift(stack[iz],shift = shift_coord[iz-1])
        stack[iz] = shifted_frame

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


def cross_corr_shift_frame(im_1, im_2, container_1 = None, container_2 = None, container_inv = None, up_rate= None):
    '''
    calculate cross-correlation based shift between two images
    assumption: the two images are in same size.
    precise to single pixels
    '''
    N, M  = im_1.shape
    hy = N//2
    hx = M//2
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
    phase_spec = F_prod/np.absolute(F_prod+1.0e-06) # phase_spec

    container_inv(phase_spec)
    corr_spec = np.abs(container_inv.get_output_array())
    shy, shx = np.unravel_index(np.argmax(corr_spec), (N,M)) # find the maximum index of normalized correlation matrix

    if shy > hy:
        shy = -(N-shy)
    if shx > hx:
        shx = -(M-shx)
    if up_rate is None:
        return shy, shx, ft_1, ft_2 # this can be used for the initial guess, the Fourier transform product is also returned
    else:
        # upsampling
        phase_N = _phase_construct_([(shy-2)*up_rate, (shy+2)*up_rate+1], [-hy, N-hy], N*up_rate, forward=False, shift_c = True)
        phase_M = _phase_construct_([-hx, M-hx], [(shx-2)*up_rate, (shx+2)*up_rate+1], M*up_rate, forward=False, shift_r = True)

        corr_spec_nbhd = np.abs(np.matmul(np.matmul(phase_N, phase_spec), phase_M)) # cross correlation in the neighborhood of (x0,y0)
        iny, inx = np.unravel_index(np.argmax(corr_spec_nbhd), corr_spec_nbhd.shape)

        shy = (iny-2*up_rate)/up_rate + shy
        shx = (inx-2*up_rate)/up_rate + shx
        return shy, shx, corr_spec_nbhd

def cross_corr_shift_stack(stack_1, stack_2=None, up_rate = 2, adj_ref = False, verbose = False):
    '''
    calculate the cross correlation
    Added: subpixel precision
    '''
    nz, ny, nx = stack_1.shape
    hy = ny//2
    hx = nx//2
    shift_coord = []
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)
    N = ny
    M = nx

    if stack_2 is None: # align the stack to itself, every frame to its adjacent frame
        ref_frame = stack_1[0]
        for ii in range(nz-1):
            y0, x0, ft_1, ft_2= cross_corr_shift_frame(ref_frame, stack_1[ii+1], container_1, container_2, container_invx)
            F_prod = np.conj(ft_1)*ft_2
            print("Initial guess:", y0, x0)
            # construct the phase matrix with smaller dimensions
            #phase_N = _phase_construct_([y0-2*up_rate, y0+2*up_rate+1], ny, ny*up_rate, forward=False)
            phase_N = _phase_construct_([(y0-2)*up_rate, (y0+2)*up_rate+1], [-hy, ny-hy], ny*up_rate, forward=False, shift_c = True)
            phase_M = _phase_construct_([-hx, nx-hx], [(x0-2)*up_rate, (x0+2)*up_rate+1], nx*up_rate, forward=False, shift_r = True)
            #phase_M = _phase_construct_(nx, [x0-2*up_rate, x0+2*up_rate+1], nx*up_rate, forward=False)

            corr_spec_nbhd = np.matmul(np.matmul(phase_N, F_prod), phase_M) # cross correlation in the neighborhood of (x0,y0)
            iny, inx = np.unravel_index(np.argmax(np.abs(corr_spec_nbhd)), corr_spec_nbhd.shape)

            shy = (iny-2*up_rate)/up_rate+y0
            shx = (inx-2*up_rate)/up_rate+x0 # the real shift in original pixel

            if verbose:
                print("slice ", ii, '-->', shy, shx)
            shift_coord.append([-shy, -shx])
            # then we have to shift im 2 by (-shy, -shx)
            shifted_frame = interpolation.shift(stack_1[ii+1],shift = [-shy, -shx])
            stack_1[ii+1] = shifted_frame
            if adj_ref:
                ref_frame = shifted_frame


    else: # align one stack to the other
        for ii in range(nz):
            y0, x0, ft_1, ft_2= cross_corr_shift_frame(stack_1[ii], stack_2[ii], container_1, container_2, container_invx)
            F_prod = np.conj(ft_1)*ft_2
            print("Initial guess:", y0, x0)

            # construct the phase matrix with smaller dimensions
            phase_N = _phase_construct_([(y0-2)*up_rate, (y0+2)*up_rate+1], [-hy, ny-hy], ny*up_rate, forward=False, shift_c = False)
            phase_M = _phase_construct_([-hx, nx-hx], [(x0-2)*up_rate, (x0+2)*up_rate+1], nx*up_rate, forward=False, shift_r = False)

            corr_spec_nbhd = np.matmul(np.matmul(phase_N, F_prod), phase_M) # cross correlation in the neighborhood of (x0,y0)
            iny, inx = np.unravel_index(np.argmax(np.abs(corr_spec_nbhd)), corr_spec_nbhd.shape)

            shy = (iny-2*up_rate)/up_rate+y0
            shx = (inx-2*up_rate)/up_rate+x0 # the real shift in original pixel


            if verbose:
                print("slice ", ii, '-->', shy, shx)
            shift_coord.append([-shy, -shx])
            shifted_frame = interpolation.shift(stack_2[ii],shift = [-shy, -shx])
            stack_2[ii] = shifted_frame

    return shift_coord



def cross_coord_shift_huge_stack(tif_handle, crop_ratio, n_cut = 5, up_rate = 2, close_handle = True):
    '''
    self-alignment of a very long stack.
    crop ratio: how much to crop in each dimension
    instead of loading the entile huge stack, we just pass the handle of the stack to the function.
    WARNING: Don't forget to close the handle!
    '''
    nz, ny, nx = tif_handle.series[0].shape
    sub_size = nz // n_cut + 1 # thenumber of slices in each substack
    print(sub_size)
    hy = ny // 2
    hx = nx // 2
    coord_all = []
    if crop_ratio is None:
        pass # this should be changed
    elif np.isscalar(crop_ratio):
        # crop x and y by the same fraction
        sy = int(crop_ratio*hy)
        sx = int(crop_ratio*hx)
    else:
        sy = int(crop_ratio[0]*hy)
        sx = int(crop_ratio[1]*hx)

    ss_0 = tif_handle.asarray()[:sub_size].astype('float64') # the first substack
    sub_0 = ss_0[:, hy-sy:hy+sy, hx-sx:hx+sx] # crop the stack to reduce the size
    shift_coord = cross_corr_shift_stack(sub_0, None, up_rate, adj_ref = False, verbose = True )
    coord_all = coord_all + shift_coord
    shift_stack(ss_0, shift_coord)
    sec_start = sub_size
    for sub_z in range(1, n_cut):
        ss_new = tif_handle.asarray()[sec_start:sec_start+sub_size]
        sub_new = ss_new[:, hy-sy:hy+sy, hx-sx:hx+sx]
        shift_coord = cross_corr_shift_stack(sub_0, sub_new, up_rate, adj_ref = True, verbose = True)
        shift_stack(ss_new, shift_coord)
        sub_0 = sub_new
        coord_all = coord_all + shift_coord

    if close_handle:
        tif_handle.close()

    return coord_all
