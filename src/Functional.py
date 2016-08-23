"""
Created on 08/22/16 by Dan. 
Calculates correlations between individual cells, create the connectivity map
"""
import numpy as np
import scipy.linalg.svd as SVD
import sklearn



class functional():
    """
    Predicts functional connectivities of 
    """
    def __init__(self, dph):
        self.dph = dph
        self.dff_data = np.load(dph)
    
        


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
        
    