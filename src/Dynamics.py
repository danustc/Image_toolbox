"""
Created by Dan Xie on 08/15/2016 
Dynamics.py: takes the extracted cell information, calculate dynamics in it. 
"""

import ntpath
import glob
import numpy as np
from scipy import linalg
from Preprocess import Drift_correction
from scipy.linalg import svd as SVD # import SVD algorithm
from common_funcs import circs_reconstruct



def z2t_convert(dph, file_flag, dims, z_frame, fmt = '02d'):
    """
    input: data path, file_flags for identification, 
    Convert selected z_frame of z-stacks into t-stacks
    dims: the ny, nx dimensions 
    tp_data: npz, with key names 'z_<number of frame>'
    """
    
    TP_list = sorted(glob.glob(dph, + '*file_flag'+ '*.npz')) # sorted by name
    NT = len(TP_list)
    clean_stack = np.zeros(dims, NT) # creat a zero-stack 
    for tp_name in TP_list:
        strip_name = ntpath.split(tp_name)[-1][:-4]  # the stripped name of the file
        strip_number = int(strip_name.split('_')[-1]) # get the time point number 
        tp_data = np.load(tp_name)
        key_z = 'z_'+ format(z_frame, fmt) # construct key_z for cellular te 
        blobs_zframe = tp_data[key_z] # take out the frame from the list 
        clean_stack[strip_number] = circs_reconstruct(dims, blobs_zframe) 
    """
    So, here we get a stack of reconstructed blobs. Let's check whether there are any empty frames: 
    
    """
    return clean_stack

 

class Dynamics(object):
    def __init__(self, TS_data):
        self.ts_data = TS_data # Is it OK to always keep the time series inside the memory? 
        
        
    def subset_TS_construction(self, subset):
        """
        input: a subset of the cell (represented by their coordinates or indices? )
        question: How to take out the subset?  
        """
        self.current_set = subset
        self.set_dynamics = self.ts_data[subset,:]
        
        
    def cell_selection(self):
        """
        input: cell coordinate in the order of [z, y, x] 
        output: The time_series 
        """
        pass
    
    
    def feature_extract(self, t_level = 3):
        """
        extract the features of the dynamics, up to the t_level th singular value
        """
        mtx = self.set_dynamics
        U, s, V = SVD(mtx) # calculate SVD
        
        U_principal = U[:t_level, :] # the first t_level rows
        V_principal = V[:, :t_level] # the first t_level columns
        s_principal = s[:t_level] # the first t_level singular values
        
        # next, let's do some projection 
        C_principal = np.dot(U_principal, np.diag(s_principal))
        
        return C_principal, V_principal # return the coefficients and singular vectors.

        
        
    