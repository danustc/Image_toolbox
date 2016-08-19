"""
Created by Dan on 08/15/2016, a test of data processing pipeline
Last update: 08/19/2016
"""
import ntpath
import os
import glob
import numpy as np


from Cell_extract import Cell_extract
from Preprocess import Deblur, Drift_correction
import tifffunc
#------------------------------------------Small functions----------------------------------

def path_leaf(path):
    """
    A tiny function for splitting filename from a long path, always the last layer, folder or not
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
    # done with path_leaf.


def number_strip(name_str, delim = '_', ext = True):
    """
    Strip the number from the filename string.
    Untested.
    """
    if(ext):
        khead = name_str.split('.')[0]
    else:
        khead = name_str
    klist = khead.split(delim) 
    knum = [km for km in klist if km.isdigit()]
    korder =knum[-1] # usually the last one represent 
    # how tor
    
    return korder

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
    def __init__(self, folder_path, tpflags = 'TP', offset = 1):
        self.work_folder = folder_path # folder path should be a folder
        self.tif_list = glob.glob(self.work_folder+ '*'+ tpflags +'*.tif')
        self.tif_list.sort(key = os.path.getmtime)
        self.tp_flag = -np.ones(len(self.tif_list)).astype('uint16') # all zero, if any time point is converted, then the element is set to True.
        self.dims = None  
        name_base = path_leaf(self.tif_list[0])[:-4] 
        parse_name = name_base.split('_') # get an array of list that 
        self.prefix_z = ''.join(parse_name[:-1]) + '_' # the prefix of the filename 
        self.offset = offset 
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
    
    
    def zstack_prepro(self, list_num = 0, deblur = 30, align = True): 
        """
        preprocess single zstack.
        list_num: the number in the tif_list  
        deblur: non-positive --- no deblur; >0 --- use as the Gaussian filter width
        align: drift align or not?
       
        """
        zs_name = self.tif_list[list_num]
        print(self.tp_flag[list_num])
        postfix_num = format(self.tp_flag[list_num], '03d') # take out the time point number 
        
        if(deblur > 0):
            prefix = 'db_'
            z_DB = Deblur(zs_name, sig = 30) # deblur
            z_dbstack = z_DB.stack_high_trunc() # return a new stack with inplane-background subtracted
            if(self.dims is None):
                self.dims = z_DB.px_num # here we get self.dims 
        else:
            prefix = ''
            z_dbstack = np.copy(tifffunc.read_tiff(zs_name)).astype('float64')

        if(align):
            prefix += 'al_'
            ofst = self.offset
            z_DC = Drift_correction(z_dbstack)
            z_newstack = z_DC.drift_correct(offset=ofst, ref_first = False)
        else:
            z_newstack = z_dbstack


        if(prefix != ''):
            z_writename = self.work_folder + self.prefix_z+prefix + postfix_num+'.tif'
            tifffunc.write_tiff(z_newstack,z_writename)

        z_CE = Cell_extract(z_newstack) 
        z_CE.stack_blobs(diam = 6)
        z_CE.stack_signal_integ()
        z_CE.save_data_list(self.work_folder+self.prefix_z+postfix_num) # save as npz
        self.pro_flag[list_num] = True
        
        return postfix_num # just for information, this returning is useless.
        # done with zstack_prepro
        
    def zstack_tseries(self, deblur = 30, align = True):
        """
        preprocess all the single zstacks. Should t-alignment be carried out here?
        0. Regardless of the orders in self.tp_flag prepreocess all the zstacks
        1. Reconstruct t-stacks based on the TP number, do the drift correction, realign as t-stacks. (must save the number of drifts.)
        2. Correct the cell positions accordingly, resave as '.npz'. (That part is kinda tricky.) 
        
        """
        for iflag in np.arange(self.n_TP):
            postfix_num = self.zstack_prepro(iflag, deblur, align) # process all the 
            print("Processed z point:", postfix_num)
            
        
        print("All done.")


    
    def z2t_construct(self):
        """
        test test.
        """
        pass 
    
    
    
#--------------------------------the counterpart for tstacks---------------------------------    
    
    
    
class pipeline_tstacks(object):
    """
    Still not quite sure if I should merge the two pipeline classes into one.
    """
    def __init__(self, folder_path, zp_flags = 'ZP'):
        self.work_folder = folder_path # folder path should be a folder
        
        tif_list = glob.glob(self.work_folder+ '*'+ zp_flags +'*.tif')
        if(not tif_list):
            print("Error! The selected folder has no required files.")
        else:
            # This part is almost parallel to the z-stack preprocessing case.
            
            tif_list.sort(key = os.path.getmtime)
            self.zp_flag = -np.ones(len(tif_list)).astype('uint16') # all zero, if any time point is converted, then the element is set to True.
            self.dims = None  
            name_base = path_leaf(tif_list[0])[:-4] 
            parse_name = name_base.split('_') # get an array of list that 
            self.prefix_t = ''.join(parse_name[:-1]) + '_' 
            self.tif_list = tif_list
            
            iflag = 0     
            for ts_name in self.tif_list:
                ts_tail = path_leaf(ts_name)[:-4] 
                ZP_number = int(ts_tail.split('_')[-1]) # get the time point number 
                self.zp_flag[iflag] = ZP_number
                iflag += 1 
                
            if(np.any(self.zp_flag<0) == True): 
                print("File name mistake. please check! ")
            else:
                print("Initialization completed! All the time-point numbers are extracted.")
                print("t-stack orders:", self.zp_flag) 
                self.n_ZP = len(self.zp_flag) # number of time points
                self.pro_flag = np.zeros(self.n_ZP).astype('bool') 
      
    
    
    def tstack_prepro(self, list_num = 0, deblur = 30, align = True, ext_all=True):
        """
        This part is similar to the zstack_prepro one in the first class.
        list_num: the number in the tif-list 
        deblur: whether background subtraction should be performed first? If deblur > 0, yes. 
        align: whether drift correction should be performed? 
        ext_all: do cell extraction from only the first slice or all the slices? 
        """
        ts_name = self.tif_list[list_num]
        print(self.zp_flag[list_num])
        postfix_num = format(self.zp_flag[list_num], '03d') # take out the time point number 
        
        if(deblur>0):
            prefix = 'db_'
            t_DB = Deblur(ts_name, sig = deblur) # deblur
            t_dbstack = t_DB.stack_high_trunc() 
            if(self.dims is None):
                self.dims = t_DB.px_num # he
        else:
            prefix = ''
            t_dbstack = np.copy(tifffunc.read_tiff(ts_name)).astype('float64')
            # directly load
        if(align):
            prefix += 'al_'
            t_DC = Drift_correction(t_dbstack)
            t_newstack = t_DC.drift_correct(offset=0, ref_first=True)
        else:
            t_newstack = t_dbstack
        
        if(prefix != ''):
            t_writename = self.work_folder + self.prefix_t+prefix + postfix_num+'.tif'
            tifffunc.write_tiff(t_newstack,t_writename)
        
        # ---------------Time for cell extraction! --------------------
        np_fname = self.work_folder+self.prefix_t+ prefix + postfix_num
        t_CE = Cell_extract(t_newstack) 
        if(ext_all):
            t_CE.stack_blobs(diam = 6)
            t_CE.stack_signal_integ()
            t_CE.save_data_list(np_fname) 
        else: # only extract cells in the first slice and assume that they persist in the rest
            # update on 08/19: save npz instead of npy.
            
            blob_time_stack = t_CE.stack_signal_propagate(0)
            np.savez(np_fname, **blob_time_stack)
        # save as npz
        self.pro_flag[list_num] = True
        
        return postfix_num # just for information, this returning is useless.
        # done with zstack_prepro 
    

            
    def tstack_zseries(self, deblur = 30, align = True, ext_all = False):
        """
        preprocess all the single zstacks. Should t-alignment be carried out here?
        0. Regardless of the orders in self.tp_flag prepreocess all the zstacks
        1. Reconstruct t-stacks based on the TP number, do the drift correction, realign as t-stacks. (must save the number of drifts.)
        2. Correct the cell positions accordingly, resave as '.npz'. (That part is kinda tricky.) 
        
        """
        for iflag in np.arange(self.n_ZP):
            postfix_num = self.tstack_prepro(iflag, deblur, align, ext_all)
            print("Processed time point:", postfix_num)
            
        
        print("All done.")
    
                
                
            
#             tifffunc.write_tiff(z_dc, zp_file[:-4]+'_pp.tif')    
        
        