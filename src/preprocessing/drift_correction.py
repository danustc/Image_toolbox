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
import tifffile as tf
import psutil
from src.visualization import brain_navigation

package_path_win  ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
global_datapath_win  = 'D:/Data/2018-08-23/B2_TS/\\'
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/Image_labs/'
global_datapath_ptb = '/media/sillycat/DanData/Jul19_2017_A2/A2_TS/'


# ----------------- Here are some preparation functions----------------

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
    w, h = raw_dataset.size # width and height
    print(w,h)
    pw, ph = patch_size

    if seek_mode == 'center':
        # just crop the center patch from the original image
        ileft = int((w-pw)//2)
        iupper = int((h-ph)//2)

    else:
        # find the optimal patch
        raw_dataset.seek(0)
        slice_0 = np.asarray(raw_dataset)
        print(slice_0.shape)
        _, rcs = patch_finding.patch_opt(slice_0)
        iupper = rcs[0] # row start
        ileft = rcs[1]  # column start


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
    filtered_stack = np.tile(hanning_2d, (NZ,1,1))*stack
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
    '''
    Open a stack, take the shifted coordinates and correct the stack on file.
    '''
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

# -------------------------------- Below is a class for drift correction ---------------
class DC_pipeline(object):
    '''
    This is a mini-pipeline for drift corrections
    '''
    def __init__(self, path):
        self.path = path
        self.stack_preparation()

    def reload_path(self. new_path):
        self.path = new_path
        self.stack_preparation()

    def stack_preparation(self):
        cropped_stack = stack_crop(self.path, seek_mode = 'opt')

        self.hf_stack = stack_hanning(cropped_stack) # hanning-filtered

    def drift_correct(self, new_path = None):
        self.shift_coord = correlation.cross_corr_stack_self(self.hf_stack)
        shift_stack_onfile(self.path, self.shift_coord, new_path)


# ---------------------------------------Below is the main function for test.-----------------------------------

def main():
    # OK the shift-on-site problem also got solved.
    folder_list = glob.glob(global_datapath_ubn+"*slice.tif")
    print(folder_list[0])
    cropped_stack = stack_crop(folder_list[0], seek_mode = 'opt')
    shift_c = correlation.cross_corr_stack_self(cropped_stack)
    print(shift_c)
    shift_stack_onfile(folder_list[0], shift_c, partial = True, srange = (0,3))


if __name__ == '__main__':
    main()
