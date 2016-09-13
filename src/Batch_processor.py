"""
Created by Dan on 08/02/16
For group processing
Last update: 09/13/16
"""

import glob
import ntpath
import tifffunc
import numpy as np
from Preprocess import Drift_correction, Deblur
from Cell_extract import Cell_extract
from Redundancy import Redundant_removal 


def group_redund_rm(datapath, dims = [1000, 1000]):
    """
    Remove redundancy from a group of .npz files 
    """
#     pass
    rrm = Redundant_removal(datapath, dims)
    rrm.zs_construct()
    rrm.corrmap_stack()
    corr_map = rrm.corr_map
    return corr_map
    
    


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
        
        
# class working_folder(object):
#     """
#     provide a folder wrapper for the working_folder, which allows batch processing
#     created on 08/15/16. 
#     """
#     def __init__(self, dph):
#         self.dph = dph
#         self.wfolder = path_leaf(self.dph)
#           
#     """
#     More to be filled here.
#     """    
#     
