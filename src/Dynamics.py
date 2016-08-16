"""
Created by Dan Xie on 08/15/2016 
Dynamics.py: takes the extracted cell information, calculate dynamics in it.
Updated by Dan on 08/16/2016 
"""

import ntpath
import glob
import numpy as np
from scipy import linalg
from Preprocess import Drift_correction
from scipy.linalg import svd as SVD # import SVD algorithm
from common_funcs import circs_reconstruct

 

class Dynamics(object):
    """
    Purpose: load one slice-all the time points
    No image operation, all based on .npz operations.   
    Recognize all the cell in the same position, extract its time train
    
    """

    def __init__(self, TS_data, dims):
        """
        TS_data: the blob values of 
        """
        self.ts_data = TS_data 
        self.dims = dims# Is it OK to always keep the time series inside the memory? 
        
        
        
    def ts_refs(self):
        """
        This one calculates all 
        
        
        """
        
        
        
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

        
        
    