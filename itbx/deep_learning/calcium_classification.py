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
global_training_path = '/home/sillycat/Programming/Python/data_test/'

smooth = 1.


def create_training_data(dff_train, label_train):
    pass






def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)  + smooth) # I don't quite understand why the dice coef is desined as this.


def model_generation(NT = 1790):
    '''
    Generate a training model
    NT: the number of time points
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



