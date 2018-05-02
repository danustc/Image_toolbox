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
    phase_spec = F_prod/np.absolute(F_prod) # phase_spec
    container_inv(F_prod)
    corr_spec = np.abs(container_inv.get_output_array())
    print(corr_spec[:3, :3])
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
        # OK this passes test.



def cross_corr_shift_self(stack, up_rate = 2, adj_ref = False, verbose = True):
    '''
    Align a stack to itself. If adj_ref is True, then align each slice to the neighboring slice preceeding it.
    Added: subpixel precision
    '''
    nz, ny, nx = stack.shape
    hy = ny//2
    hx = nx//2
    shift_coord = []
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)

    ref_frame = stack[0]
    for ii in range(nz-1):
        shy, shx  = cross_corr_shift_frame(ref_frame, stack[ii+1], container_1, container_2, container_invx,up_rate)[:2]
        if verbose:
            print("slice ", ii+1, '-->', shy, shx)
        shift_coord.append([-shy, -shx])
        # then we have to shift im 2 by (-shy, -shx)
        if adj_ref: # if the stack is aligned in the adjacent mode, them each slice should be updated in place; otherwise we can just return the shift coordinates.
            shifted_frame = interpolation.shift(stack[ii+1],shift = [-shy, -shx])
            ref_frame = shifted_frame
            stack[ii+1] = shifted_frame

    return shift_coord # Hmmmm, this is much nicer.


def cross_corr_shift_two(stack_1, stack_2, up_rate = 2, verbose = True, inplace_correction = False):
    '''
    Align stack_2 to stack_1
    '''
    nz, ny, nx = stack_1.shape
    hy = ny//2
    hx = nx//2
    shift_coord = []
    container_1 = pyfftw_container(ny, nx)
    container_2 = pyfftw_container(ny, nx)
    container_invx = pyfftw_container(ny, nx, bwd = True)


    for ii in range(nz):
        shy, shx = cross_corr_shift_frame(stack_1[ii], stack_2[ii], container_1, container_2, container_invx, up_rate)[:2]

        if verbose:
            print("slice ", ii, '-->', shy, shx)
        shift_coord.append([-shy, -shx])
        if inplace_correction:
            shifted_frame = interpolation.shift(stack_2[ii],shift = [-shy, -shx])
            stack_2[ii] = shifted_frame

    return shift_coord # Yay! This is much nicer.



def cross_coord_shift_huge_stack(huge_stack, crop_ratio = 0.8, n_cut = 5, up_rate = None):
    '''
    self-alignment of a very long stack.
    crop ratio: how much to crop in each dimension
    instead of loading the entile huge stack, we just pass the handle of the stack to the function.
    WARNING: Don't forget to close the handle!
    '''
    nz, ny, nx = huge_stack.shape
    if (nz&n_cut ==0):
        sub_size = nz // n_cut
    else:
        sub_size = nz // n_cut + 1 # thenumber of slices in each substack

    substack_posit = np.arange(1, n_cut+1)*sub_size
    substack_posit[-1] = nz # make sure that the last position is consistent with the stack size
    hy = ny // 2
    hx = nx // 2
    coord_all = []
    if crop_ratio is None:
        cstack = huge_stack
    elif np.isscalar(crop_ratio):
        # crop x and y by the same fraction
        sy = int(crop_ratio*hy)
        sx = int(crop_ratio*hx)
        cstack = huge_stack[:,hy-sy:hy+sy, hx-sx:hx+sx]
    else:
        sy = int(crop_ratio[0]*hy)
        sx = int(crop_ratio[1]*hx)
        rg_y = np.arange(hy-sy, hy+sy)
        rg_x = np.arange(hx-sx, hx+sx)
        cstack = huge_stack[:,rg_y, rg_x]

    sub_0 = cstack[:sub_size] # crop the stack to reduce the size
    shift_coord = cross_corr_shift_self(sub_0, up_rate, adj_ref = False, verbose = True )
    shift_stack(huge_stack[:sub_size], shift_coord) # self-align the first substack
    coord_all = coord_all + shift_coord
    sub_0 = cstack[:sub_size] # This might be redundant 
    for sub_z in range(n_cut-1):
        sec_start = substack_posit[sub_z]
        sec_end = substack_posit[sub_z+1]
        sub_new = cstack[sec_start:sec_end]
        shift_coord = cross_corr_shift_two(sub_0, sub_new, up_rate, verbose = True, inplace_correction = False)
        shift_stack(huge_stack[sec_start:sec_end], shift_coord)
        coord_all = coord_all + shift_coord

    return np.array(coord_all)




def main():
    pass
