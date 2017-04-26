"""
Last update: 04/25/2016
Test affine transformation
"""


import os
from src.preprocessing.z_dense import z_dense_ref, z_dense_construct
import numpy as np
import matplotlib.pyplot as plt
from src.Cell_extract import Cell_extract
import src.preprocessing.tifffunc as tifffunc
import src.preprocessing.Affine as Affine
from src.visualization.brain_navigation import slice_display

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'


def dumb1():
    '''
    0. load two slices, one reference and one rotated
    1. extract all the cells from each slice and save the output
    2. Apply the affine transformation matrices onto the rotated slice
    3. Display the two groups of cells on the same figure to compare the transformation.
    '''

    ref_im = tifffunc.read_tiff(global_datapath+'ref_crop.tif')
    rot30_im = tifffunc.read_tiff(global_datapath+'rot30_crop.tif')
    rotpt_im = tifffunc.read_tiff(global_datapath+'rot30_crop_pt.tif')
    compstack = np.array([ref_im,rot30_im, rotpt_im]).astype('float64')

    CE = Cell_extract(compstack)
    CE.stack_blobs(msg = True)
    #CE.save_data_list(global_datapath+'ref_rot')
    coord_list= CE.get_coordinates()
    print(coord_list.keys())
    coord_ref = coord_list['s_000'] # flip the original coordinates
    coord_rot = np.fliplr(coord_list['s_001'])
    coord_pts = coord_list['s_002']
    af_mat, af_vec = Affine.aff_read(global_datapath + 'sliceReg.txt')
    af_mat = af_mat[0]
    af_vec = af_vec[0]

    raf_mat, raf_vec = Affine.reverse_trans(af_mat, af_vec)
    afc_rot = np.fliplr(Affine.pixel_transform(coord_rot, raf_mat, raf_vec))
    fig_comp= slice_display([coord_ref,afc_rot ])
    print("Cells plotted!")
    fig_comp.savefig(global_datapath+'ref_display')
#     fig_rot.savefig(global_datapath+'rot_display')


if __name__ == '__main__':
    dumb1()# 
