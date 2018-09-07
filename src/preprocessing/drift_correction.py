'''
New trial of drift correction
'''
import pyfftw
import numpy as np
from scipy.ndimage import interpolation
import tifffile as tf
import correlation
import patch_finding

package_path_win  ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
global_datapath_win  = 'D:/Data/2018-08-23/\\'


def shift_stack(stack, shift_coord):
    '''
    shift a stack via interpolation
    '''
    nz, _, _ = stack.shape
    for iz in range(nz):
        shifted_frame = interpolation.shift(stack[iz],shift = shift_coord[iz-1])
        stack[iz] = shifted_frame


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
    folder_list = glob.glob(data_rootpath+"/Aug*/*TS/\\")
    for data_path in folder_list:
        print(data_path)
        img_path = glob.glob(data_path + 'rg*.tif')
        for stack_name in img_path:
            stack = tf.imread(stack_name)
            slice_0 = stack[0] # the first stack
            patch = 

