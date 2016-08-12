"""
Created by Dan on 08/02/16
For group processing
Last update: 08/10/16
"""

import glob
import tifffunc
import numpy as np
from Preprocess import Drift_correction, Deblur
from Cell_extract import Cell_extract


def group_alignment(datapath, nameflag = 'TS', ofst = 1, mfit = 7):
    # align all th
    # prerequisite: all slices should be deblurred inplane
    tiff_list = glob.glob(datapath + '*' + nameflag+'*.tif')
    for tiff_name in tiff_list: 
        n_base = tiff_name[:-4] # remove '.tif' extension
        new_stack = np.copy(tifffunc.read_tiff(n_base))
        Drift_C = Drift_correction(new_stack, mfit)
        a_stack = Drift_C.drift_correct(offset = ofst)
        n_out = n_base+'_a'
        tifffunc.write_tiff(a_stack[ofst:].astype('uint16'), n_out)
        

def group_deblur_inplane(datapath, nameflag = 'TS*', sig= 30, n_apdx = '_db', overwrite = True):
    """
    group_deblur_inplane, regardless of aligned or not 
    """
    raw_list = glob.glob(datapath + '*' + nameflag + '*.tif')
    for raw_name in raw_list:
        n_base = raw_name[:-4]
        DB = Deblur(impath = n_base, sig)
        DB.stack_high_trunc(adjacent= False, wt=0)
        if(overwrite == False):
            DB.write_stack(n_apdx)
        else:
            DB.write_stack(None)
    print("The stack has been deblurred. ")        
            
            
            
            
def group_deblur_cross():
    # to be filled later 
    pass


def group_cell_extract(datapath, name_flag = 'TS*'):
    stack_list = glob.glob(datapath+'*'+ name_flag + '*.tif')
    for stk_name in stack_list:
        n_base = stk_name[:-4]
        im_stack = tifffunc.read_tiff(n_base)
        CE = Cell_extract(im_stack)
        CE.stack_blobs()
        
        