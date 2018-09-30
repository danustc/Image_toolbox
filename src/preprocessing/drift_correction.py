'''
New trial of drift correction
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
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
from src.visualization import brain_navigation

package_path_win  ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
global_datapath_win  = 'D:/Data/2018-08-23/B2_TS/\\'
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/Image_labs/'
global_datapath_ptb = '/media/sillycat/DanData/Jul19_2017_A2/A2_TS/'


def stack_crop(raw_stack_path, patch_size = (512,512), seek_mode = 'center'):
    '''
    Update on 09/28/2018: Split this into two functions
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
    cropped_stack = np.zeros((raw_dataset.n_frames, ph, pw)) # pay attention to the stack size!

    crop_canvas = (ileft, iupper, iright, ilower)
    for ii, page in enumerate(ImageSequence.Iterator(raw_dataset)):
        cropped_stack[ii] = page.crop(crop_canvas)
        if(ii%100 == 0):
            print(ii, '------------------Loaded. ')

    raw_dataset.close() # remember to close the file everytime you open it!

    return cropped_stack


def stack_hanning(stack):
    '''
    Create a hanning filter and apply it on each slice of the stack
    '''
    NZ, NY, NX = stack.shape
    # create a 2d Hanning filter
    hann_w = signal.hann(NX)
    hann_h = signal.hann(NY)
    hanning_2d = np.outer(hann_h, hann_w)
    hstack = np.tile(hanning_2d, (NZ,1,1))
    filtered_stack = stack*hstack
    return filtered_stack



def shift_stack(stack, shift_coord):
    '''
    shift a stack via interpolation
    '''
    nz, _, _ = stack.shape
    for iz in range(nz):
        shifted_frame = interpolation.shift(stack[iz],shift = shift_coord[iz-1])
        stack[iz] = shifted_frame

    return stack

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



def noise_simulation(NZ, NY, NX, model = 'gauss', mean = 160, sig = 100):
    '''
    simulate a pure noisy stack.
    '''
    if model =='gauss':
        noise_stack = np.random.normal(loc = mean, scale = sig, size = (NZ, NY, NX))
    elif model == 'poisson':
        noise_stack = np.random.poisson(lam = mean, size = (NZ, NY, NX))

    return noise_stack



def stack_simulation(img, shift_list, noise_model = 'gauss', noise_level = 0.1):
    '''
    simulate a noisy, drifted image with raw image as the seed; if img is None, then simulate a stack with noise only.
    '''
    NZ = len(shift_list) + 1
    sig_mean, sig_std = np.mean(img), np.std(img)
    if img is None:
        NY, NX = NR
    else:
        NY, NX = img.shape

    stack = np.zeros((NZ, NY, NX))
    stack[0] = img
    for ii in range(1, NZ):
         stack[ii] = interpolation.shift(img,shift_list[ii-1])

    if noise_level > 0:
        noise_stack = noise_simulation(NZ, NY, NX, noise_model, mean = noise_level*sig_mean, sig = sig_std*noise_level)
        print(noise_stack.max())
        stack = stack + noise_stack

    return stack



# ---------------------------------------Below is the main function for test.-----------------------------------

def main():
    folder_list = glob.glob(global_datapath_ubn+"*slice.tif")
    for data_path in folder_list:
        print(data_path)
        sample_stack= tf.imread(data_path).astype('float64')
        noise_slice = np.random.poisson(size = (512,512), lam = 0.5)
        noise_slice = noise_slice**2
        ft_raw = correlation.fft_image(sample_stack[1])
        sample_hanning = stack_hanning(sample_stack)
        ft_han = correlation.fft_image(sample_hanning[1])
        ft_noise= correlation.fft_image(noise_slice)

        fig_raw = brain_navigation.slices_compare(sample_stack[1], np.log(ft_raw), 'Raw image', 'Fourier Transform')
        fig_han = brain_navigation.slices_compare(sample_hanning[1], np.log(ft_han), 'Hanning-filtered image', 'Fourier Transform')
        fig_noise = brain_navigation.slices_compare(noise_slice, np.log(ft_noise), 'Simulated noise', 'Fourier Transform')
        fig_raw.savefig('raw_fft')
        fig_han.savefig('han_fft')
        fig_noise.savefig('noise_fft')



        shift_coord = correlation.cross_corr_stack_self(sample_stack, pivot_slice = 0)
        np.savetxt(global_datapath_ubn + 'drift_raw.txt', shift_coord, delimiter = '\t', fmt = '%d')
        vraw = shift_coord.var(axis = 0 )
        mraw = np.abs(shift_coord).sum(axis = 0)
        shift_coord = correlation.cross_corr_stack_self(sample_hanning, pivot_slice = 0)
        vhan = shift_coord.var(axis = 0 )
        mhan = np.abs(shift_coord).sum(axis = 0)
        np.savetxt(global_datapath_ubn + 'drift_han.txt', shift_coord, delimiter = '\t', fmt = '%d')
        print("mean of raw:", mraw)
        print("mean of han-filtered:", mhan)
        #print(shift_coord)
        #shift_stack_onfile(data_path, shift_coord, new_path = global_datapath_win+ '/test.tif')

if __name__ == '__main__':
    main()
