"""
Created by Dan on 10/10/2016. Load a densely imaged stack (npz file, with all the slices corrected and the cells extracted )
"""

import numpy as np
from Cell_extract import Cell_extract
from Preprocess import Drift_correction
from scipy.stats.mstats_basic import threshold


class z_dense_ref(object):
    """
    load a densely labeled stack 
    """
    
    def __init__(self, densefile, dims):
        z_dense = np.load(densefile) 
        z_keys = z_dense.keys()
        z_keys.sort()
        nz  = len(z_keys)
        ny, nx = dims
        
        for iz in np.arange(nz):
            """
            clear the diameter column of each frame and use it as a counter.
            """
            zk  = z_keys[iz]
            z_dense[zk][:,3] = 0 
        
        
        
        self.z_dense = z_dense # so here it is automatically sorted 
#         self.zd_stack = np.zeros(nz, ny, nx) # have a "virtual stack"  
        
    
    
    def zd_construct(self):
        """
        reconstruct a new densely-imaged stack
        """
        
    def _red_detect_(self, nslice = 0, thresh = 1.0):
        """
        detect redundancy centered around the slice n.
        thresh: threshold for redundancy detection  
        """
        zk_1 = 's_' + format(nslice, '03d')
        zk_2 = 's_' + format(nslice+1, '03d') 
        
        zf_1 = self.z_dense[zk_1]
        zf_2 = self.z_dense[zk_2]
        
        # extract the y and x coordinates 
        y1 = zf_1[:,0]
        x1 = zf_1[:,1]
        
        y2 = zf_2[:,0]
        x2 = zf_2[:,0]
        
        # create a meshgrid 
        [YC, YR] = np.meshgrid(y2, y1)
        [XC, XR] = np.meshgrid(x2, x1) 
        
        
        dist_block = np.sqrt((YC-YR)**2 + (XC-XR)**2)
        red_pair = np.where(dist_block <= thresh) # find out where  
        
        ind1 = red_pair[0] # the indices in the first frame 
        ind2 = red_pair[1] # the indices in the second frame 
        
        zf_1[ind1, 3] +=1 
        zf_2[ind2, 3] +=zf_1[ind1, 3]   # mark the fold of redundancy. I think this is a smart step. 
        
        
        
        
        
        
        