'''
New trial of drift correction
'''
import numpy as np
from scipy.ndimage import interpolation
from scipy import signal
import correlation
import patch_finding
from PIL import Image, ImageSequence, ImageStat
import glob
import matplotlib.pyplot as plt
import tifffile as tf
import psutil

package_path_win  ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
global_datapath_win  = 'D:/Data/2018-08-23/B2_TS/\\'
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/cmtk_images/'
global_datapath_ptb = '/media/sillycat/DanData/Jul19_2017_A2/A2_TS/'


def sample_data_generation(img, n_slice, n_shifts, sample_path):
    '''
    generate a sample data_stack from a raw image.
    '''
    print(n_shifts)
    new_stack = np.tile(img, (n_slice, 1, 1)) # create a stack
    ph, pw = img.shape
    hann_w = signal.hann(pw)
    hann_h = signal.hann(ph)
    hanning_2d = np.outer(hann_h, hann_w)
    for ii in range(1, n_slice):
        new_stack[ii] = interpolation.shift(img, shift = n_shifts[ii-1])*hanning_2d

    tf.imsave(sample_path, new_stack)
    return new_stack

def stack_preparation(raw_stack_path, patch_size = (512,512), seek_mode = 'center', ):
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
    hann_w = signal.hann(pw)
    hann_h = signal.hann(ph)
    hanning_2d = np.outer(hann_h, hann_w)
    cropped_stack = np.zeros((raw_dataset.n_frames, ph, pw)) # pay attention to the stack size!

    crop_canvas = (ileft, iupper, iright, ilower)
    for ii, page in enumerate(ImageSequence.Iterator(raw_dataset)):
        cropped_stack[ii] = page.crop(crop_canvas)*hanning_2d
        if(ii%100 == 0):
            print(ii, '------------------Loaded. ')

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

def shift_stack_onfile(fpath, shift_coord, new_path = None, partial = False, srange = (0, 500)):
    if partial:
        print("Only save part of the stack.")
        with tf.TiffFile(fpath) as tif:
            raw_img = tif.asarray()[srange[0]:srange[1]]
            shift_coord = shift_coord[srange[0]:srange[1]]
            tif.close()
    else:
        raw_img = tf.imread(fpath)
    n_slice, ny, nx = raw_img.shape
    for ii in range(n_slice):
        frame = raw_img[ii]
        raw_img[ii] = interpolation.shift(frame, shift = shift_coord[ii])

    if new_path is None:
        tf.imsave(fpath, raw_img)
    else:
        tf.imsave(new_path, raw_img)



def main():
    #folder_list = glob.glob(global_datapath_ubn+"/Aug*.tif")
    #folder_list = glob.glob(global_datapath_ptb+"/dup*.tif")
    folder_list = glob.glob(global_datapath_win+"/*.tif")
    for data_path in folder_list:
        print(data_path)
        img = tf.imread(data_path)
        #sample_stack = sample_data_generation(img[0], 4, shift_list, 'test.tif')
        cropped_stack = stack_preparation(data_path)
        #print(cropped_stack.dtype)
        shift_coord = correlation.cross_corr_stack_self(cropped_stack, pivot_slice = 0)
        #print(shift_coord)
        shift_stack_onfile(data_path, shift_coord)

if __name__ == '__main__':
    main()
