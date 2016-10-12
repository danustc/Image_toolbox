"""
Created by Dan on 10/10/2016. Load a densely imaged stack (npz file, with all the slices corrected and the cells extracted )
"""

import numpy as np
from Cell_extract import Cell_extract
from Preprocess import Drift_correction
from scipy.stats.mstats_basic import threshold
from linked_list import Simple_list


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
        
            
        # select those with markers > 0 and markers < 0  
        new_idx = (ind1 ==0 )
        pool_new = ind1[new_idx]
        pool_new_cov = ind2[new_idx]
        
        pool_exist = ind1[~new_idx]
        pool_exist_cov = ind2[~new_idx]
        
        n_new = len(pool_new)
        n_exist = len(pool_exist)
        
        
        for n_ind in pool_new:
            # build the new keys 
            pr_key = str(nslice) + '_' + str(n_ind)
            new_sl = Simple_list(nslice) # create a simple list with z_marker = nslice   
            new_sl.add([n_ind, zf_1[n_ind, 4]]) 
            
            self.redundancy_pool[pr_key] =   
            
        
        
        zf_1[ind1, 3] +=1 
        zf_2[ind2, 3] += zf_1[ind1, 3]  # mark the fold of redundancy. I think this is a smart step. 
        
        
        return ind1, ind2  # return the indices which indicates the marked positions of redundancy 
        # end of paired redundancy detection 
        
        
    def stack_red_detect(self):
        """
        detect the whole stack's redundancy and correct them.
        steps: 
        0. Create an empty dictionary to store the link list 
        1. compare the first two frames (z0, z1) and detect the redundancy, return the indices 
        2. create a couple of link lists that have two nodes, set all the "updated" status as False.
        3. compare the next two frames (z1, z2) and detect the redundancy again. If the cells of z1 are unmarked in the last comparison, 
        initialize a new linklist; otherwise find the identity key (should be a string) and append the frame number to the old list. 
        """
        
        self.redundancy_pool = {}  # this should be initialized first, so that it can be updated in every round of detection 
        
        ind_init, ind_next = self._red_detect_(0, thresh = 1.0)
        
        for cell in ind_init: 
            cell_key = 'z0_'+ str(cell)
            
            new_SL = Simple_list(z_marker = 0)
            new_node = [cell, self.z_dense] 
            new_SL.add([cell, ])
        
        