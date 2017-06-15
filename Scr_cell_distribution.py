"""
Last update: 06/15/2017
A global view of cell distribution in the whole set of data
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

def ZD_visualization(dph_im):
    '''
    visualize a Z-stack
    dph_im: the path to .npz file
    '''
    fname_stem = os.path.splitext(dph_im)[0] # split the filename stem
    ZDR = z_dense_construct(dph_im)
    return




def dumb1(ariz_list = [], rota_list = []):
    '''
    0. Load two slices, first one is the matched slice in the ZD stack, second one is the second slice in the ts stack.
    1. Load all the retrieved cells
    2. Apply the affine transformation inversely to the cells tracked from the T-slice
    3. Cross align the positions of the T-cells with those in the Z-stacks.
    '''
    # read the Z-slices and the t-slices
    ZD_stack = 'A1_FB_ZD.tif'


    zd_file = 'A1_FB_ZD.npz'
    dims = [732, 908]
    ZDR = z_dense_construct(global_datapath+zd_file)
    ZD_red = z_dense_ref(ZDR, dims)
    zstack_3d = ZD_red.stack_red_detect()
    print(zstack_3d.shape)
    fig3d = stack_display(zstack_3d, cl = 'b')
    ax =fig3d.gca()
    for ariz in ariz_list:
        for rota in rota_list:
            ax.view_init(ariz, rota)
            fig3d.tight_layout()
            fig_str = 'view'+ str(int(ariz))+ '_'+ str(int(rota))
            fig3d.savefig(global_datapath+fig_str)




if __name__ == '__main__':
    ariz_list = [0, 45, 90]
    rota_list = [0, 30, 60, 90, 120, 150]
    dumb1(ariz_list, rota_list)# 
