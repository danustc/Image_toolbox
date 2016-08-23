"""
Created on 08/22/16 by Dan. 
Calculates correlations between individual cells, create the connectivity map
"""
import numpy as np
import scipy.linalg.svd as SVD
import glob
from string_funcs import *


class functional():
    """
    Predicts functional connectivities of 
    """
    def __init__(self, dph):
        self.dph = dph
        self.ts_data  = dict()
        self.coord = dict()
    
    def __load__(self):
        """
        load all the k-value data into the 
        """
        f_list = glob.glob(self.dph+'*') 
        for zs_file in f_list:
            pl = path_leaf(zs_file)
            znum = number_strip(pl, delim = '_', ext = False)
            kwd = 'z_'+str(znum).zfill(3)
            data_z = np.load(zs_file)
            self.coord[kwd] = data_z['xy']
            self.ts_data[kwd] = data_z['data']
        # done with loading data.        
    
    
    def correlation_map(self):    
        pass


    def feature_extract(self, t_level = 3):
        """
        extract the features of the dynamics, up to the t_level th singular value
        """
        dff_mtx = self.dff_data
        U, s, V = SVD(dff_mtx) # calculate SVD
        
        U_principal = U[:t_level, :] # the first t_level rows
        V_principal = V[:, :t_level] # the first t_level columns
        s_principal = s[:t_level] # the first t_level singular values
        
        # next, let's do some projection 
        C_principal = np.dot(U_principal, np.diag(s_principal))
        
        return C_principal, V_principal # return the coefficients and singular vectors.
        
    