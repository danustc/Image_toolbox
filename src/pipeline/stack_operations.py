'''
Based on the basic functions in tifffunc.py, Here are some stack operations that can substitute messy macros in ImageJ.
Created by Dan Xie on 04/06/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import src
import numpy as np
import tifffunc as tf
import pyfftw
import glob
import src.shared_funcs.string_funcs as sfc

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

def T2Z_sample(wfolder,n_slice, zpflag = 'ZP', save_flag='tz'):
    '''
    select the n_sliceth frame from each ZP stack, order them by the filename, and save as a new stack. If any number is lacking, insert an empty frame or a frame with all-negative numbers.
    '''
    if(wfolder[-1] != '/'):
        ins = '/*'
    else:
        ins = '*'
    zp_list = glob.glob(wfolder+ins+zpflag+'*.tif')
    nzp = len(zp_list)
    # first, check completeness of the zp points.
    zp_ind = []
    for zp_fname in zp_list:
        naked_name = sfc.path_leaf(zp_fname)
        z_slice = sfc.number_strip(naked_name, delim = '_', ext = True)
        zp_ind.append(z_slice)
    # now, sort and check the completeness/redundancy
    zarg_sort=np.argsort(zp_ind)
    zp_list=zp_list[zarg_sort]
    zp_ind = zp_ind[zarg_sort] # sort through the list 
    # 1. check redundancy
    if(nzp!=len(set(zp_ind))):
        print("There are redundancies ")
        zp_ind = list(set(zp_ind))
    # 2. check any missing point
    zrange = zp_ind[-1] - zp_ind[0] + 1
    zstart = zp_ind[0]
    if (zrange > nzp):
        print("The zp files are incomplete.")
    stack_represent = []
    for iz in np.arange(zrange):
        '''
        fill up the stack_represent
        '''
        zcount = iz + zstart
        if(zcount in zp_ind):
            # the slice exists. 
            z_slice = tf.read_tiff(zp_list[iz], n_slice)
            stack_represent.append(z_slice)
        else:
            stack_represent.append(0) #append 0 to the list 

    ny, nx = z_slice.shape # there must be at list real z_slice.


    return stack_represent

def main():
    '''
    a test function,test stack splitting functions
    '''
    impath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    ZD_name= 'A1_FB_ZD.tif'
    TS_name= 'TS_folder/TS_ZP_9.tif'
    ZD_stack = tf.read_tiff(impath+ZD_name)
    '''
    TS_slice= tf.read_tiff(impath+TS_name, 1) # take 5 slices
   1 loc_stack = frame_locate(ZD_stack, 36,verbose = True)
    tf.write_tiff(loc_stack,impath+'loc.tif')
    n_loc = loc_stack.shape[0]
    TS_dup = duplicate_tiff(TS_slice, n_loc)
    tf.write_tiff(TS_dup, impath+'ts_dup.tif')
    '''
    loc_stack = tf.read_tiff(impath+'loc.tif')
    TS_dup = tf.read_tiff(impath+'ts_dup.tif')
    corr_trace = correlate_trace(loc_stack, TS_dup)
    print(corr_trace)

    print("Finished!")


def scratch():
    '''
    test the translation
    '''
    impath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    TS_dup = tf.read_tiff(impath + 'ts_dup.tif')
    ref_frame = TS_dup[0]
    cor_frame = np.roll(TS_dup[1], 10,axis = 0)
    cor_frame = np.roll(cor_frame, 10,axis = 1)
    tf.write_tiff(ref_frame, impath + 'ref.tif')
    tf.write_tiff(cor_frame, impath + 'trans_10.tif')




if __name__=='__main__':
    scratch()


