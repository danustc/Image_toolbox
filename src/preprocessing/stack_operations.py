'''
Based on the basic functions in tifffunc.py, Here are some stack operations that can substitute messy macros in ImageJ.
Created by Dan Xie on 04/06/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import src
import numpy as np
import src.shared_funcs.tifffunc as tf
from PIL import Image


def binning(stack, nb = 2):
    '''
    bin the original stacks into nbxnb.
    '''
    nz, ny, nx = stack.shape
    print(nz, ny, nx)
    hy = ny//nb
    hx = nx//nb
    samplegrid_y = np.arange(hy)*nb
    samplegrid_x = np.arange(hx)*nb
    bstack = np.zeros((nz,hy,hx))

    for ii in range(nb):
        for jj in range(nb):
            temp = stack[:,ii:hy*nb+ii:nb,jj:hx*nb+jj:nb ]
            bstack +=temp
    bstack /=(nb*nb)
    return bstack


def duplicate_tiff(frame, ndup):
    '''
    create a tiff stack from one duplicated slice
    '''
    dup_stack = np.tile(frame, [ndup,1,1])
    return dup_stack



def crop_tiff(imstack,positions ):
    '''
    crop a tiff image
    imstack is already an np array
    '''
    yi = positions[0]
    yf = positions[2]+ yi
    xi = positions[1]
    xf = positions[3]+ xi
    cr_stack = imstack[:,yi:yf, xi:xf]
    return cr_stack

def frame_locate(im_ref,z_init, ref_step = 1.0, z_range=4.0, verbose = False):
    '''
    im_ref: a reference stack
    assume the reference stack always covers the range 0 --- nslice
    z_init: integer or floating point
    ref_step: the z-step of the densely-labeled stack.
    z_range: searching range
    '''
    nslice = im_ref.shape[0]*ref_step
    ind_boundary = int(z_range/ref_step)
    ind_range = int(z_init/ref_step)+np.arange(-ind_boundary,ind_boundary)
    lb = (ind_range>=0)
    rb = (ind_range<nslice)
    ind_range = ind_range[np.logical_and(lb,rb)]
    if verbose:
        print("The selected range:", ind_range*ref_step) # This is the range in the real z instead of the slice indices.
    locate_substack = im_ref[ind_range]
    return locate_substack



def stack_split(raw_stack, nsplit, dph = None):
    '''
    Split a substack from a raw stack and save if dph is not None.
    '''
    substack = raw_stack[nsplit]

    if dph is None:
        return substack
    else:
        tf.write_tiff(substack, dph)

def stack_propagate(raw_stack, mode = 'a'):
    '''
    do some operation on a raw stack. Output: a processed slice.
    modes:  a---average
            s--- sum
            other modes to be filled up later .
    '''
    ns = raw_stack.shape[0]
    if mode =='a':
        return raw_stack.sum(axis = 0)/ns
    elif mode =='s':
        return raw_stack.sum(axis = 0)
    # done with stack_propagate

def stack_section(raw_stack, px_position, view = 'z'):
    '''
    take out a section from a stack along a plane perpendicular to the view direction.
    '''
    nz, ny, nx =raw_stack.shape
    if (view == 'z'):
        zslice =np.min([int(np.round(px_position)), nz-1])
        im_section = raw_stack[zslice]
    elif (view == 'y'):
        yslice = np.min([int(np.round(px_position)), ny-1])
        im_section = raw_stack[:, yslice, :]
    else:
        xslice = np.min([int(np.round(px_position)), nx-1])
        im_section = raw_stack[:,:,xslice]
    return im_section

def stack_padding(raw_stack, dest_size):
    '''
    pad a new stack with zeros
    '''
    new_stack = np.zeros(dest_size)
    nz, ny, nx = raw_stack.shape
    dz, dy, dx = dest_size
    hz = (dz-nz)//2
    hy = (dy-ny)//2
    hx = (dx-nx)//2
    new_stack[hz:hz+nz, hy:hy+ny, hx:hx+nx] = raw_stack
    return new_stack

def reorient_tiff_general(imstack, axis = 0):
    '''
    general imstack rotation
    '''
    rot_stack = np.rot90()



def reorient_tiff_RAS(imstack, fname):
    '''
    Reorient the image stack in the RAS coordinate system.
    Note that the original image.
    '''
    # nz, nx, ny = imstack.shape
    rot_stack = []
    for zslice in imstack:
        rot_stack.append(np.rot90(zslice, k=3))

    rot_stack = np.array(rot_stack)
    tf.write_tiff(rot_stack, fname)


def dynamic_reduce(imstack, nbit = 8):
    '''
    reduce the dynamic range of given images
    '''
    red_stack = imstack/(2**nbit)
    return np.round(red_stack).astype('uint16') # return as uint16


def stack_global_thresholding(stack, nsig = 3):
    s_mean = stack.mean()
    s_std = stack.std()
    th_stack = np.copy(stack)
    th_stack[stack>s_mean+nsig*s_std] = s_mean+nsig*s_std
    return th_stack
