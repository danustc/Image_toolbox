"""
Last update: 04/25/2016
Test affine transformation
"""


import os
from src.preprocessing.z_dense import z_dense_ref, z_dense_construct
import numpy as np
import matplotlib.pyplot as plt
from src.Cell_extract import Cell_extract
import src.preprocessing.tifffunc as tf
import src.preprocessing.Affine as Affine
from src.visualization.brain_navigation import slice_display,stack_display

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'


def dumb1(ariz_list = [], rota_list = []):
    '''
    0. Load two slices, first one is the matched slice in the ZD stack, second one is the second slice in the ts stack.
    1. Load all the retrieved cells
    2. Apply the affine transformation inversely to the cells tracked from the T-slice
    3. Cross align the positions of the T-cells with those in the Z-stacks.
    '''
    # read the Z-slices and the t-slices
    TS_slice9 = 'TS_folder/rg_A1_FB_TS_ZP_9.tif'
    TS_slice14 = 'TS_folder/rg_A1_FB_TS_ZP_14.tif'
    ZD_stack = 'A1_FB_ZD.tif'
    zstep = 4
    Z_slices = tf.read_tiff(global_datapath+ZD_stack, np.array([9,14])*zstep)
    T_slice9 = tf.read_tiff(global_datapath+TS_slice9, 1)
    T_slice14 = tf.read_tiff(global_datapath + TS_slice14, 1)

    # load the cell extraction data 

    TS14 = np.load(global_datapath+'TS_14.npz')
    coord_14, f_14 = TS14['xy'], TS14['data'] # split the coordinates and data
    print("coord:", coord_14.shape)
    # do the similar thing for slice 9
    TS9= np.load(global_datapath+'TS_9.npz')
    coord_9, f_9 = TS9['xy'], TS9['data'] # split the coordinates and data

    zd_file  = 'A1_FB_ZD.npz'
    dims = [732, 908]
    zd_coord = np.load(global_datapath+zd_file)
    ZDR = z_dense_construct(global_datapath+zd_file)
    ZD_red = z_dense_ref(ZDR, dims)
    zstack_3d = ZD_red.stack_red_detect()
    print(zstack_3d.shape)
   # next, let's load the affine transformation and apply them on the TS stack.
    afm_9, afv_9 = Affine.aff_read(global_datapath+'ts2zd_9.txt', True)
    afm_14, afv_14 = Affine.aff_read(global_datapath+'ts2zd_14.txt', True)
    rfm_09, rfv_09 = Affine.reverse_trans(afm_9, afv_9)
    rfm_14, rfv_14 = Affine.reverse_trans(afm_14, afv_14)
    afc_09= np.fliplr(Affine.pixel_transform(np.fliplr(coord_9), rfm_09, rfv_09))
    afc_14 = np.fliplr(Affine.pixel_transform(np.fliplr(coord_14), rfm_14, rfv_14))

    # afc09, afc14 are the new coordinates after Affine Transformation
    zf_3d = ZD_red.frame_zalign(afc_09, z_init = 36, thresh = 4.0)
    print(zf_3d.shape)
    zd_09 = zd_coord['s_036']
    zd_14 = zd_coord['s_056']
    fig_09 = slice_display([zd_09, afc_09], title = 'Slice 9')
    fig_14 = slice_display([zd_14, afc_14], title = 'Slice 14')
    fig_09.savefig(global_datapath+'match_slice09')
    fig_14.savefig(global_datapath+'match_slice14')



if __name__ == '__main__':
    dumb1()# 
