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
    # do the similar thing for slice 9
    TS9= np.load(global_datapath+'TS_9.npz')
    coord_14, f_14 = TS9['xy'], TS9['data'] # split the coordinates and data

    zd_file = 'A1_FB_ZD.npz'
    dims = [732, 908]
    ZDR = z_dense_construct(global_datapath+zd_file)
    ZD_red = z_dense_ref(ZDR, dims)
    zstack_3d = ZD_red.stack_red_detect()
    print(zstack_3d.shape)
    fig3d = stack_display(zstack_3d, cl = 'g')
    ax =fig3d.gca()
    for ariz in ariz_list:
        for rota in rota_list:
            ax.view_init(ariz, rota)
            fig3d.tight_layout()
            fig_str = 'view'+ str(int(ariz))+ '_'+ str(int(rota))
            fig3d.savefig(global_datapath+fig_str)


if __name__ == '__main__':
    ariz_list = [0, 45, 90]
    rota_list = [0, 30, 60, 90]
    dumb1(ariz_list, rota_list)# 
