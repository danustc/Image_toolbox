# coding: utf-8
'''
Dan's first convolutional neural network test!
'''
from __future__ import print_function
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import backend as K
import os
import glob
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
from skimage.io import imsave, imread
global_training_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

smooth = 1.

def create_training_data(rt_path, tc_flags = None, n_row = 800, n_col = 1000):
    '''
    create training data from the existing tiff stacks, save as npy
    '''
    im_lists = glob.glob(rt_path+'*'+tc_flags + '*.tif')
    for im_name in im_lists:
        im_stack = tf.read_tiff(im_name)
        nz, ny, nx = im_stack.shape
        x_pad = n_col - nx
        y_pad = n_row - ny
        im_padded = np.pad(im_stack, [(0,0), (y_pad, 0), (x_pad, 0)], mode = 'minimum')
        np.save(os.path.splitext(im_name), im_padded)
    print("Saving completed.")


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)  + smooth) # I don't quite understand why the dice coef is desined as this.


def model_generation(im_row, im_col):
    '''
    Generate a training model
    '''
    inputs = Input((im_row, im_col, 1))
    conv1 = Conv2D(32,(3,3), activation='relu', padding = 'same')(inputs)
    conv1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(conv1)

    conv_end = Conv2D(1, (1,1), activation = 'sigmoid')(pool1)

    model = Model(inputs = [inputs], outputs = [conv_end] )
    model.compile(optimizer = Adam(lr = 1.0e-05), loss = -dice_coef, metrics = [dice_coef])

    return model

def train_and_predict(train_datapath, train_mask_datapath, test_datapath, test_id_datapath):
    '''

    '''



