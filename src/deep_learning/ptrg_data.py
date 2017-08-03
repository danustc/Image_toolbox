# coding: utf-8
'''
Dan's first convolutional neural network test!
This is the pattern recognition part of the cnn-based pattern recognition.
'''
from __future__ import print_function
import os
import glob
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
from collections import  deque
global_training_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

smooth = 1.
srow = 96
scol = 96


def flag_match(im_list, mask_flag):
    '''
    test if the im_lists contains a whole list of raw data and masks
    '''
    n_file = len(im_list)
    mask_ind = [i for i, s in enumerate(im_list) if mask_flag in s] # this is very pythonic way to extract indices
    data_ind = list(set(range(n_file))-set(mask_ind))
    mask_list = [im_list[mi] for mi in mask_ind] # the mask_list
    data_list = [im_list[di] for di in data_ind] # the data_list

    dq_data = deque()
    dq_mask = deque()
    for dt_name in data_list:
        dt_base = os.path.basename(dt_name).split('.')[0] # strip out the basename without extension 
        ms_base = dt_base+'_'+mask_flag


    return valid_datalist, valid_masklist



def create_training_data(rt_path, tc_flags, mask_flags, n_row = 800, n_col = 1000):
    '''
    create training data from the existing tiff stacks, save as npy
    '''
    im_lists = glob.glob(rt_path+'*'+ tc_flags + '*.tif')
    n_training = len(im_lists)/2 # the total number 
    im_data = np.ndarray((n_training,s_row, s_col), dtype = np.uint8)
    im_mask = np.ndarray((n_training, s_row, s_col), dtype = np.uint8)


    ii = 0
    for im_name in im_lists:
        im_stack = imread(im_name, as_grey = True)
        nz, ny, nx = im_stack.shape
        x_pad = n_col - nx
        y_pad = n_row - ny
        im_padded = np.pad(im_stack, [(0,0), (0, y_pad), (0, x_pad)], mode = 'minimum') # first, pad the image to create uniform sizes
        im_mask[ii]= resize(im_padded, (srow, scol), preserve_range = True, mode = 'constant')
        ii +=1


    np.save(os.path.splitext(im_name), im_padded)
    print("Saving completed.")


