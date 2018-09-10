'''
New trial of drift correction
'''
import numpy as np
from scipy.ndimage import interpolation
import correlation
import patch_finding
from PIL import Image, ImageSequence, ImageStat
import glob
import matplotlib.pyplot as plt

package_path_win  ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
global_datapath_win  = 'D:/Data/2018-08-23/B2_TS/\\'
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/cmtk_images/'


def stack_preparation(raw_stack_path, patch_size = (512,512), seek_mode = 'center', padding_width = 10):
    '''
    prepare a raw stack for drift alignment
    instead of loading the whole full raw stack, just open its path and seek one slice
    patch_size: (pw, ph), opposite to the convention of Python-array dimensions.
    # return a cropped stack
    '''
    # 1. take the first slice and calculate patch
    raw_dataset = Image.open(raw_stack_path)
    slice_0 = raw_dataset.seek(0)
    w, h = raw_dataset.size # width and height
    pw, ph = patch_size
    ileft = int((w-pw)//2)
    iupper = int((h-ph)//2)
    iright = ileft + pw
    ilower = iupper + ph
    cropped_stack = np.zeros((raw_dataset.n_frames, ph+2*padding_width, pw+2*padding_width)) # pay attention to the stack size!

    crop_canvas = (ileft, iupper, iright, ilower)
    for ii, page in enumerate(ImageSequence.Iterator(raw_dataset)):
        cropped_stack[ii] = np.pad(page.crop(crop_canvas), (padding_width, padding_width),  mode = 'constant')

    raw_dataset.close() # remember to close the file everytime you open it!

    return cropped_stack


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
    #folder_list = glob.glob(global_datapath_ubn+"/Aug*.tif")
    folder_list = glob.glob(global_datapath_win+"/*5.tif")
    for data_path in folder_list:
        print(data_path)
        cropped_stack = stack_preparation(data_path)
        print(cropped_stack.dtype)
        correlation.cross_corr_stack_self(cropped_stack, pivot_slice = 10)

if __name__ == '__main__':
    main()
