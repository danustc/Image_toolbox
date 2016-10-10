"""
Created by Dan on 10/10/2016. Load a densely imaged stack (npz file, with all the slices corrected and the cells extracted )
"""

import numpy as np
from Cell_extract import Cell_extract
from Preprocess import Drift_correction


class z_dense_ref(object):
    """
    load a densely labeled stack 
    """
    
    def __init__(self, densefile, dims):
        z_dense = np.load(densefile) 
        z_keys = z_dense.keys().sort()
        
        
        
        