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
import time
from PIL import Image

global_datapath = '/home/sillycat/Programming/Python/data_test/'
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

def sample_from_refstack(stack_ref, sample_range, pxl_ref, pxl_sample,  rotmat, rshift):
    '''
    stack_ref: the reference stack, i.e., the Z-brain.
    sample_range: the range of sampling, in pixels.
    pxl_ref: the pixel size of the reference stack
    pxl_sample: the pixel size of the sample stack.
    r_shift: the translation vecter from the lab frame to the image frame, unit of pixel.
    '''
    sx, sy, sz = sample_range

    hx = sx//2
    hy = sy//2
    hz = sz//2
    print("Half range:", hx, hy, hz)
    ix = (np.arange(sx)-hx)*pxl_sample[0]
    iy = (np.arange(sy)-hy)*pxl_sample[1]
    iz = (np.arange(sz)-hz)*pxl_sample[2]
    [MY, MZ, MX]=np.meshgrid(iy, iz, ix) # the grid coordinates of the image stacks
    fmz = np.ndarray.flatten(MZ)
    fmy = np.ndarray.flatten(MY)
    fmx = np.ndarray.flatten(MX)
    flat_imgcoord = np.c_[fmx, fmy, fmz]
    flat_labcoord = (np.dot(flat_imgcoord, rotmat))/pxl_ref+rshift # the coordinates in the lab frame in the unit of pixels. 
    inter_signal = np.empty(sx*sy*sz) # initialize an empty array
    ii = 0
    for coord in flat_labcoord:
        inter_signal[ii] = trilinear_interpolation(stack_ref, coord)
        ii+=1
        if ii%10000 == 0:
            print("----------Finished %d out of %d calculations.------------"%(ii,sx*sy*sz))

    sample_value = np.reshape(inter_signal, [sz, sy, sx]) #note that the order of indices in 3D python array.
    return sample_value



def sample_to_refstack(substack, ref_range, pxl_sample, pxl_ref, rotmat, rshift):
    '''
    A reference stack with size known, a registered chunk of image, which should be filled into the empty reference frame.
    '''
    ref_stack = np.zeros(ref_range) # create an empty stack
    rz, ry, rx = ref_range
    sz, sy, sx = substack.shape
    hz = sz//2
    hx = sx//2
    hy = sy//2
    idx_offset = np.array([hx,hy,hz])
    ix = (np.arange(sx)-hx)*pxl_sample[0]
    iy = (np.arange(sy)-hy)*pxl_sample[1]
    iz = (np.arange(sz)-hz)*pxl_sample[2]
    [MY, MZ, MX]=np.meshgrid(iy, iz, ix) # the grid coordinates of the image stacks
    fmz = np.ndarray.flatten(MZ)
    fmy = np.ndarray.flatten(MY)
    fmx = np.ndarray.flatten(MX)
    flat_imgcoord = np.c_[fmx, fmy, fmz]
    flat_labcoord = np.round((np.dot(flat_imgcoord, rotmat))/pxl_ref+rshift) # the coordinates in the lab system, unit pixel
    flat_imgcoord_approx = np.dot((flat_labcoord-rshift)*pxl_ref, rotmat.T)/pxl_sample   # convert back to the image coordinate
    ii = 0
    for coord in flat_imgcoord_approx:
        indx = flat_labcoord[ii,[2,1,0]].astype('int') # nz, ny, nx
        img_coord = coord+idx_offset
        if (indx[0] < rz and indx[1] < ry and indx[2] < rx):
            if(sx > img_coord[0] and sy > img_coord[1] and sz > img_coord[2]):
                ref_stack[indx[0], indx[1], indx[2]] = trilinear_interpolation(substack, img_coord)
        ii+=1
        if ii%10000 == 0:
            print("--------Finished %d out of %d calculations. ------"%(ii, sx*sy*sz))
    return ref_stack
    # [MY, MZ, MX] = np.meshgrid(iy,iz, ix)

# +++++++++++++++++++++++++++++++++++++++++++++ The Class ++++++++++++++++++++++++++++++++++
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
    def convert2lab(self, shift = None):
        '''
        convert the imaging frame into lab frame. Should have a reverse version.
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
        if shift is not None:
            flat_coord +=shift # If the origin is shifted from lab frame to the image frame.
        return flat_labcoord
        # this is the stupid method


    def convert2img_trilinear_interpolation(self, pxl_lab, lab_shape, lab_shift, verbose = True):
        '''
        lab_pxl: the pixel size in the lab frame
        given a stack coordinate in the lab frame and a known image stack, calculate the representation of the imagein the lab frame.
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
def scr_padding():
    '''
    pad all the ZD stacks with zeros and resave
    '''
    ZD_list = glob.glob(global_datapath+'Dec*/*ZD*.tif')
    #padded_size = [101, 804, 1020]
    print(ZD_list)
    for ZD_file in ZD_list:
        basename = ''.join(ZD_file.split('/')[-2].split('_'))
        ZD_stack = tf.read_tiff(ZD_file)
        zz, zy, zx = np.where(ZD_stack==0)
        ZD_stack[zz,zy,zx] = np.std(ZD_stack)/2.0 + np.random.randn(len(zz))*10
        #ZD_padded = stack_padding(ZD_stack, padded_size)
        tf.write_tiff(ZD_stack, global_datapath+basename+'.tif')
        #ZD_front = stack_split(ZD_padded, np.arange(101), global_datapath+basename + '_f.tif')
        #ZD_back= stack_split(ZD_padded, np.arange(101)+1, global_datapath+basename + '_b.tif')

def main():
    # test slice splitting function
    #refstack = tf.read_tiff(regist_path+'refbrain/Nov012016B3ZDref.tif')
    #reorient_tiff_RAS(refstack, regist_path+'refbrain/Nov012016B3ZDRASref.tif')
    data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'refbrain/rfp_temp.tif'
    im_path = 'GCamp_reg1.tif'

    #ref_stack = tf.read_tiff(data_path+ref_path)
    substack = tf.read_tiff(data_path+im_path)
    rm_yaxis = rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    #sample_value = sample_from_refstack(ref_stack, sample_range, pxl_lab, pxl_img, rm_yaxis, origin_shift)
    ref_empty = sample_to_refstack(substack, ref_range, pxl_img, pxl_lab, rm_yaxis, origin_shift)
    tf.write_tiff(ref_empty, data_path + 'GCamp_resample.tif' )
    print("done!")

if __name__ =='__main__':
    scr_padding()
