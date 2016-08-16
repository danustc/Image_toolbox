"""
Updated by Dan on 08/16/2016.
"""
import sys
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import os
import glob
from Preprocess import Deblur, Drift_correction
from Cell_extract import Cell_extract
from Pipeline import pipeline_zstacks, pipeline_tstacks




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

    """
    OK this part already works. 
    """ 
    hroot = 'D:\Data/'
    abspath = os.path.abspath(hroot)
    aq_date = '/2016-03-21\\'
    fd = abspath + aq_date
    fd_list = glob.glob(fd+'*') # list all the tiff files in the folder  
    
    tpflags = 'ZP'
    
    for work_folder in fd_list:
        work_folder = work_folder + '\\'
        print(work_folder)
    
        TS_list = glob.glob(work_folder+'*'+tpflags+'*.tif')
        for tfile in TS_list:
            print(tfile)
            ZP_DB = Deblur(tfile,sig=30)
            z_new = ZP_DB.stack_high_trunc(adjacent=False)
            ZP_DC = Drift_correction(z_new, mfit = 0)
            z_dc = ZP_DC.drift_correct(offset=0, ref_first=True)
            tifffunc.write_tiff(z_dc, tfile[:-4]+'_pp.tif')
        
        
        
def dumb3():
    """
    suppose drift already corrected. Now time for real cell extraction! 
    The first part is almost identical to dumb2().
    
    """
    hroot = 'D:\Data/'
    abspath = os.path.abspath(hroot)
    aq_date = '/2016-03-21/zp_A2_4_TS1\\'
    work_folder = abspath + aq_date+'\\'
    
    
    
    print(work_folder)
    t_pipeline = pipeline_tstacks(work_folder)
    t_pipeline.tstack_zseries(deblur=30, align = True)
#     pz.zstack_prepro(0)

#     CE_dbl.stack_signal_archive()
#     CE_dbl.save_archive('arc_img_dbl')
#     n_frame = 10    
#     data_slice_2 = CE_dbl.image_signal_integ(n_frame)
#     fig2 = CE_dbl.frame_display(n_frame, False)
#     fig2.savefig('dbl_s10')
#     
#     fig3.savefig('stack_reconstruction')

if __name__ == '__main__':
    dumb3()

