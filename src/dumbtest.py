"""
Updated by Dan on 08/11/2016.
"""

import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import os
import glob
from Preprocess import Deblur, Drift_correction
from Cell_extract import Cell_extract
from Pipeline import pipeline_zstacks


def dumb1():
    """
    This program tests pipeline_zstacks 
    """
    dph = '/home/sillycat/Documents/Zebrafish/Exp_figures/'
    tpflags = 'TP'
    pz = pipeline_zstacks(dph, tpflags)
    pz.zstack_prepro(0)
    pz.zstack_tseries()


def dumb2():

    dbl_image = 'raw_image_deblur'
    
    dbl_stack = tifffunc.read_tiff(dbl_image).astype('float64')
    CE_dbl = Cell_extract(dbl_stack)
    CE_dbl.stack_blobs(diam = 6)
    CE_dbl.stack_signal_integ()
    CE_dbl.save_data_list('blobs_list')
    print(CE_dbl.bl_flag)
    
#     CE_dbl.stack_signal_archive()
#     CE_dbl.save_archive('arc_img_dbl')
#     n_frame = 10    
#     data_slice_2 = CE_dbl.image_signal_integ(n_frame)
#     fig2 = CE_dbl.frame_display(n_frame, False)
#     fig2.savefig('dbl_s10')
#     
#     fig3.savefig('stack_reconstruction')

if __name__ == '__main__':
    dumb1()

