"""
Created by Dan on 08/15/2016, a test of data processing pipeline
"""
import ntpath
import os
import glob
import numpy as np
import tifffunc


from Cell_extract import Cell_extract
from Preprocess import Deblur, Drift_correction
from sqlalchemy.testing.plugin.plugin_base import post_begin

def path_leaf(path):
    """
    A tiny function for splitting filename from a long path, always the last layer, folder or not
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# -----------------------------------------Big classes-------------------------------------------------

class pipeline_zstacks(object):
    """
    This pipeline processes all the zstacks inside a folder.
    Procedure: 
    0. glob --- find all the z-stacks in the working folder.  (completed)
    1. load a stack of tif file (in z stack) (completed)
    2. Split the file name to get time-point number, convert the original integers to 3-digit integers (completed) 
    2. Perform in-plane deblur on the z-stack, return a cleaner one (completed) 
    3. Perform Cell extraction on the deblurred z-stack, save as 'npz' (completed)
    4. reconstruct t-stacks from z-stacks (can be run from outside)
    """
    def __init__(self, folder_path, tpflags = 'TP'):
        self.work_folder = folder_path # folder path should be a folder
        self.tif_list = glob.glob(self.work_folder+ '*'+ tpflags +'*.tif')
        self.tif_list.sort(key = os.path.getmtime)
        self.tp_flag = -np.ones(len(self.tif_list)).astype('uint16') # all zero, if any time point is converted, then the element is set to True.
            
        name_base = path_leaf(self.tif_list[0])[:-4] 
        parse_name = name_base.split('_') # get an array of list that 
        self.prefix_z = ''.join(parse_name[:-1]) + '_' # the prefix of the filename 
        
        # OK, time to extract all the flags
        iflag = 0     
        for zs_name in self.tif_list:
            zs_tail = path_leaf(zs_name)[:-4] 
            TP_number = int(zs_tail.split('_')[-1]) # get the time point number 
            self.tp_flag[iflag] = TP_number
            iflag += 1 
            
        if(np.any(self.tp_flag<0) == True): 
            print("File name mistake. please check! ")
        else:
            print("Initialization completed! All the time-point numbers are extracted.")
            print("z-stack orders:", self.tp_flag) 
            self.n_TP = len(self.tp_flag) # number of time points
            self.pro_flag = np.zeros(self.n_TP).astype('bool') # an all-false array to mark t
    
    
    def zstack_prepro(self, list_num = 0): 
        """
        preprocess single zstack.
        list_num: the number in the tif_list  
        """
        zs_name = self.tif_list[list_num]
        print(self.tp_flag[list_num])
        postfix_num = format(self.tp_flag[list_num], '03d') # take out the time point number 
        
        
        z_DB = Deblur(zs_name, sig = 30) # deblur
        z_dbstack = z_DB.stack_high_trunc() # return a new stack with inplane-background subtracted


        z_CE = Cell_extract(z_dbstack) 
        z_CE.stack_blobs(diam = 6)
        z_CE.stack_signal_integ()
        z_CE.save_data_list(self.work_folder+self.prefix_z+postfix_num) # save as npz
        self.pro_flag[list_num] = True
        
        return postfix_num # just for information, this returning is useless.
        # done with zstack_prepro
        
    def zstack_tseries(self):
        """
        preprocess all the single zstacks. Should t-alignment be carried out here?
        0. Regardless of the orders in self.tp_flag prepreocess all the zstacks
        1. Reconstruct t-stacks based on the TP number, do the drift correction, realign as t-stacks. (must save the number of drifts.)
        2. Correct the cell positions accordingly, resave as '.npz'. (That part is kinda tricky.) 
        
        """
        for iflag in np.arange(self.n_TP):
            self.zstack_prepro(iflag) # process all the 
        
        print("All done.")
        
        
#         zb_list = glob.glob(self.work_folder+self.prefix_z + '*.npz') # list all the ''
        
        
        