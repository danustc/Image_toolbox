'''
Based on the basic functions in tifffunc.py, Here are some stack operations that can substitute messy macros in ImageJ.
Created by Dan Xie on 04/06/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import src
import numpy as np
import src.shared_funcs.tifffunc as tf
import pyfftw
import glob
import os
import src.shared_funcs.string_funcs as sfc

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
regist_path = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/'


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


def duplicate_tiff(frame, ndup):
    '''
    create a tiff stack from one duplicated slice
    '''
    dup_stack = np.tile(frame, [ndup,1,1])
    return dup_stack




def crop_tiff(imstack,positions, cfname):
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


def correlate_trace(stack_ref, stack_corr):
    '''
    calculate correlations between every two slices in stack_ref and stack_corr
    '''
    rz,ry,rx = stack_ref.shape
    cz,cy,cx = stack_corr.shape
    if(ry!=cy or rx!=cx):
        print("Error! The slice dimensions mismatch.")
        return
    nloop = np.min([rz,cz])
    corr_trace = np.zeros(nloop) # initialize an empty array to store the correlation functions.
    container_ref = pyfftw_container(ry,rx)
    container_cor = pyfftw_container(cy,cx) # build pyfftw objects
    container_invx = pyfftw_container(ry,rx, bwd = True)
    for ii in np.arange(nloop):
        # calculate through the stack 
        container_ref(stack_ref[ii])
        container_cor(stack_corr[ii])
        ft_ref = container_ref.get_output_array()
        ft_cor = container_cor.get_output_array()
        F_prod = np.conj(ft_ref)*ft_cor
        container_invx(F_prod)
        X_corr = np.abs(container_invx.get_output_array())
        corr_trace[ii] =  np.max(X_corr)

    return corr_trace


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


#--------------------------------Test the the stack operations---------------

def main():
    # test slice splitting function
    refstack = tf.read_tiff(regist_path+'refbrain/Nov012016B3ZDref.tif')
    reorient_tiff_RAS(refstack, regist_path+'refbrain/Nov012016B3ZDRASref.tif')


if __name__ =='__main__':
    main()
