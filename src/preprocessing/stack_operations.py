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


def rotmat_yaxis(ra):
    '''
    Calculate the rotational matrix with respect to y.
    ra: rotational angle in degrees
    '''
    rotmat = np.zeros((3,3))
    rad_ra = ra*np.pi/180.0
    rotmat[0] = np.array([np.cos(rad_ra), 0, np.sin(rad_ra)])
    rotmat[1,1] = 1.0
    rotmat[2] = np.array([-np.sin(rad_ra), 0, np.cos(rad_ra)])
    return rotmat

def resample_rot_frame(coord_ref, pxl_ref, pxl_rot, rot_mat, shift = np.zeros(3)):
    '''
    pxl_ref: the pixel size of the imaging stacks in the lab frame
    pxl_rot: the pixel size of the imaging stacks in the imaging frame
    rot_mat: the rotational matrix in the 3D space
    shift: whether the translation from ref to rot occurs
    Assume that we know the coordinates in the reference frame, this function calculates the coordinates in the rotated frame.
    '''
    coord_rot = np.dot(coord_ref*pxl_ref, rot_mat.T)/pxl_rot
    return coord_rot # the m', n', k' coordinates in the rotated frame. May not be integers

def trilinear_interpolation(stack, coord_px):
    '''
    trilinear interpolation in the 3d space.coord_px: x,y,z coordinates in the unit of pixel. Need to be permuted?
    if the coord_px exceeded the stack size, assign 0.0
    '''
    nz, ny, nx = stack.shape
    cx = coord_px[0]
    cy = coord_px[1]
    cz = coord_px[2]
    x0 = int(np.floor(cx))
    y0 = int(np.floor(cy))
    z0 = int(np.floor(cz))
    xd = cx-x0
    yd = cy-y0
    zd = cz-z0

    x1 = x0+1
    y1 = y0+1
    z1 = z0+1

    try:
        p000 = stack[z0, y0, x0]
    except IndexError:
        p000 = 0.
    try:
        p100 = stack[z0, y0, x1]
    except IndexError:
        p100 = 0.
    try:
        p010 = stack[z0, y1, x0]
    except IndexError:
        p010 = 0.
    try:
        p001 = stack[z1, y0, x0]
    except IndexError:
        p001 = 0.
    try:
        p110 = stack[z0, y1, x1]
    except IndexError:
        p110 = 0.
    try:
        p101 = stack[z1, y0, x1]
    except IndexError:
        p101 = 0.
    try:
        p011 = stack[z1, y1, x0]
    except IndexError:
        p011 = 0.
    try:
        p111 = stack[z1, y1, x1]
    except IndexError:
        p111 = 0.

    dx_y0z0 = xd*p100+(1.-xd)*p000
    dx_y1z0 = xd*p110+(1.-xd)*p010
    dx_y0z1 = xd*p101+(1.-xd)*p001
    dx_y1z1 = xd*p111+(1.-xd)*p011

    dy_z0 = yd*dx_y1z0 + (1.-yd)*dx_y0z0
    dy_z1 = yd*dx_y1z1 + (1.-yd)*dx_y0z1

    dz = zd*dy_z1 + (1.-zd)*dy_z0

    return dz



class Stack_rot_resample(object):
    def __init__(self, stack, pxl_img):
        '''
        the class of resampling between two transformed coordinate systems
        rotmat: from the lab frame to the imaging frame
        '''
        self._imstack = stack
        self._pxl = pxl_img
        self._rotmat = None

    @property
    def rotmat(self):
        return self._rotmat
    @rotmat.setter
    def rotmat(self, new_rotmat):
        self._rotmat = new_rotmat

    @property
    def imstack(self):
        return self._imstack
    @imstack.setter
    def imstack(self, new_stack):
        self._imstack = new_stack


    @property
    def pxl_img(self):
        return self._pxl
    @pxl_img.setter
    def pxl_img(self, new_pxl):
        self._pxl = new_pxl

    #----------------------------- public functions -------------------------
    def convert2lab(self):
        '''
        convert the imaging frame into lab frame.
        '''
        nz, ny, nx = self.imstack.shape
        rz = np.arange(nz)*pxl_img[2]
        ry = np.arange(ny)*pxl_img[1]
        rx = np.arange(nx)*pxl_img[0]


        [MY, MZ, MX]=np.meshgrid(ry, rz, rx) # the grid coordinates of the image stacks
        fmz = np.ndarray.flatten(MZ)
        fmy = np.ndarray.flatten(MY)
        fmx = np.ndarray.flatten(MX)
        flat_imgcoord = np.c_[fmx, fmy, fmz]
        flat_labcoord = np.dot(flat_imgcoord, self.rotmat)
        return flat_labcoord
        # this is the stupid method

    def convert2img_trilinear_interpolation(self, pxl_lab, lab_shape, lab_shift, verbose = True):
        '''
        lab_pxl: the pixel size in the lab frame
        '''
        lz, ly, lx = lab_shape
        rz = np.arange(lz)*pxl_lab[2]-lab_shift[2]
        ry = np.arange(ly)*pxl_lab[1]-lab_shift[1]
        rx = np.arange(lx)*pxl_lab[0]-lab_shift[0]
        [MY, MZ, MX]=np.meshgrid(ry, rz, rx) # the grid coordinates of the image stacks
        fmz = np.ndarray.flatten(MZ)
        fmy = np.ndarray.flatten(MY)
        fmx = np.ndarray.flatten(MX)
        flat_labcoord = np.c_[fmx, fmy, fmz]
        flat_imgcoord = np.dot(flat_labcoord, self.rotmat.T)/self.pxl_img # convert to pixel size
        flat_imgcoord[flat_imgcoord<0] = np.max(lab_shape)
        npx_lab = lab_shape.prod()
        inter_signal = np.empty(npx_lab)

        ii = 0
        for coord in flat_imgcoord:
            inter_signal[ii] = trilinear_interpolation(self.imstack, coord) # convert to pixel in the imaging frame
            ii+=1
            if ii%10000== 0:
                print("----------Finished %d out of %d calculations.------------"%(ii,npx_lab))

        lab_value = np.reshape(inter_signal, lab_shape)
        return lab_value



 #--------------------------------Test the the stack operations---------------

def main():
    # test slice splitting function
    #refstack = tf.read_tiff(regist_path+'refbrain/Nov012016B3ZDref.tif')
    #reorient_tiff_RAS(refstack, regist_path+'refbrain/Nov012016B3ZDRASref.tif')
    data_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    img_stack = tf.read_tiff(data_path+'Jun06_A2_GCDA/A2_ZD_before.tif')
    rm_yaxis = rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.785, 0.785, 2.00]
    lab_shape = np.array([100, 300, 300])
    stk_rot_sample = Stack_rot_resample(img_stack, pxl_img)
    stk_rot_sample.rotmat=rm_yaxis
    lab_value = stk_rot_sample.convert2img_trilinear_interpolation(pxl_lab, lab_shape, lab_shift = [50.0, 0.0, 20.0])
    tf.write_tiff(lab_value, data_path + 'ZD_resample.tif' )

if __name__ =='__main__':
    main()
