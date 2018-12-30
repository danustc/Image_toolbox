'''
New trial of drift correction: Hanning filter is used to eliminate non-periodicity artifact.
'''
import os.path as opath
import numpy as np
from scipy.ndimage import interpolation
from itbx.preprocessing import correlation
import patch_finding
from PIL import Image, ImageSequence, ImageStat
import glob
import tifffile as tf

global_datapath_yst = 'D:/Dan/Data_Rock/\\'
global_datapath_win  = 'E:/2018-10-11/A1_TS/\\'
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/Image_labs/'
global_datapath_ptb = '/media/sillycat/DanData/Jul19_2017_A2/A2_TS/'

# ----------------- Here are some preparation functions----------------

def stack_crop(raw_stack_path, patch_size = (512,640), seek_mode = 'center'):
    '''
    Update on 09/28/2018: Split this into two functions
    prepare a raw stack for drift alignment
    instead of loading the whole full raw stack, just open its path and seek one slice
    patch_size: (ph, pw), consistent with the convention of Python-array dimensions.
    # return a cropped stack
    '''
    # 1. take the first slice and calculate patch
    raw_dataset = tf.imread(raw_stack_path)
    n_frames, h, w = raw_dataset.shape# width and height
    ph, pw = patch_size

    if seek_mode == 'center':
        # just crop the center patch from the original image
        ileft = int((w-pw)//2)
        iupper = int((h-ph)//2)

    else:
        # find the optimal patch
        slice_0 = raw_dataset[0]
        _, rcs = patch_finding.patch_opt(slice_0, stride = 15)
        iupper = rcs[0] # row start
        ileft = rcs[1]  # column start


    iright = ileft + pw
    ilower = iupper + ph
    cropped_stack = np.zeros((n_frames, ph, pw)) # pay attention to the stack size!

    crop_canvas = (ileft, iupper, iright, ilower)
    print(crop_canvas)
    for ii, page in zip(range(n_frames), raw_dataset):
        cropped_stack[ii] = page[iupper:ilower][:,ileft:iright]
        if(ii%100 == 0):
            print(ii, '------------------Loaded. ')

    return cropped_stack


def shift_stack(stack, shift_coord):
    '''
    shift a stack via interpolation
    '''
    nz, _, _ = stack.shape
    for iz in range(nz):
        shifted_frame = interpolation.shift(stack[iz],shift = shift_coord[iz-1])
        stack[iz] = shifted_frame

    return stack


def shift_stack_onfile(fpath, shift_coord, new_path = None, partial = False, srange = (0, 500), sub = False):
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
        print("save the whole stack.")
        with tf.TiffFile(fpath) as tif:
            raw_img = tif.asarray()

    n_slice, ny, nx = raw_img.shape
    # subpixel correction interpolation needed
    for ii in range(n_slice):
        frame = raw_img[ii]
        raw_img[ii] = interpolation.shift(frame, shift = shift_coord[ii], order = 0)
        if (ii%100 == 0):
            print("Finished ", ii, "slices.")

    if new_path is None:
        tf.imsave(fpath, raw_img.astype('uint16'))
    else:
        tf.imsave(new_path, raw_img.astype('uint16'))



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
    def __init__(self, path = None):
        if path is not None:
            self.path = path
            self.stack_preparation()

    def reload_path(self, new_path):
        self.path = new_path
        self.basename = opath.basename(new_path) # the basename including tif extension
        self.dir = opath.dirname(new_path)
        self.stack_preparation()

    def stack_preparation(self, pad_width = 20):
        cropped_stack = stack_crop(self.path, seek_mode = 'center')
        print(cropped_stack.shape)
        if pad_width > 0:
            self.hf_stack = np.pad(cropped_stack, [(0,0), (pad_width,pad_width),(pad_width,pad_width)], mode = 'constant') # hanning-filtered

    def drift_correct(self, n_pivots = [50, 900, 1700], new_path = None):
        if new_path is None:
            new_path = self.dir + '/dc_' + self.basename

        if np.isscalar(n_pivots): # only do one round
            shift_coord = correlation.cross_corr_stack_self(self.hf_stack, pivot_slice = n_pivots)
            shift_stack_onfile(self.path, shift_coord, new_path, partial = False, srange = (0,200))
        else:
            # do multiple rounds of drift correction
            shift_total = np.zeros([self.hf_stack.shape[0], 2])
            for pslice in n_pivots:
                shift_coord = correlation.cross_corr_stack_self(self.hf_stack, pivot_slice = pslice)
                self.hf_stack = shift_stack(self.hf_stack, shift_coord)
                shift_total = shift_total + shift_coord

            shift_stack_onfile(self.path, shift_total, new_path, partial = False)



# ---------------------------------------Below is the main function for test.-----------------------------------

def main():
    # OK the shift-on-site problem also got solved.
    folder_list = glob.glob(global_datapath_win+  'A*21.tif')
    print(folder_list)
    PL = DC_pipeline()
    for fname in folder_list:
        PL.reload_path(fname) # stack_preparation is also done
        PL.drift_correct(n_pivots = 100)


if __name__ == '__main__':
    main()
