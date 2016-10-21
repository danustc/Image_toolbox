"""
Created by Dan on 10/10/2016. Load a densely imaged stack (npz file, with all the slices corrected and the cells extracted )
"""

import numpy as np
from linked_list import Simple_list


def z_dense_construct(zd_file):
    zd = np.load(zd_file)
    z_dense = {}
    for keys, values in zd.items():
        new_entry = values
        new_entry[:,3] = 0
        z_dense[keys] = new_entry
        
    return z_dense


class z_dense_ref(object):
    """
    load a densely labeled stack 
    """
    def __init__(self, z_dense, dims, z_step=1.0):
        z_keys = z_dense.keys()
        nz  = len(z_keys)
        self.ny, self.nx = dims
        self.z_dense = z_dense
        self.nz = nz
        self.z_step = 1.0
         

    
    def _red_detect_(self, nslice = 0, thresh = 2.0):
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
        x2 = zf_2[:,1]
        
        
        # create a meshgrid 
        [YC, YR] = np.meshgrid(y2, y1)
        [XC, XR] = np.meshgrid(x2, x1) 
        
        
        dist_block = np.sqrt((YC-YR)**2 + (XC-XR)**2)
        red_pair = np.where(dist_block <= thresh) # find out where  
        
        ind1 = red_pair[0] # the indices in the first frame 
        ind2 = red_pair[1] # the indices in the second frame 
        
            
        # select those with markers > 0 and markers < 0 
        marker_1 = zf_1[ind1, 3] 
        
        

        new_idx = (marker_1 == 0)
        pool_new = ind1[new_idx]
        pool_new_cov = ind2[new_idx]
        
        
        pool_exist = ind1[~new_idx]
        pool_exist_cov = ind2[~new_idx]
        
        n_new = len(pool_new)
        n_exist = len(pool_exist)

        print(n_new, n_exist, "nums!")
        
        for n_count in np.arange(n_new):
            # build the new keys
            # also, we need to assign each new key an identity number which is unique.  
            n_ind1 = pool_new[n_count]
            n_ind2 = pool_new_cov[n_count]
            pr_number =  nslice * 1000 + n_ind1
            pr_key = 'sl_' + str(pr_number) 
            new_sl = Simple_list(nslice) # create a simple list with z_marker = nslice   
            new_sl.add([nslice, zf_1[n_ind1, 4]]) 
            new_sl.add([nslice+1, zf_2[n_ind2, 4]])
            zf_1[n_ind1, 3] = pr_number 
            zf_2[n_ind2, 3] = pr_number
            
            self.redundancy_pool[pr_key] = new_sl   
            
            
        for n_count in np.arange(n_exist):
            # search for the existing keys 
            n_ind1 = pool_exist[n_count]
            n_ind2 = pool_exist_cov[n_count]
            pr_number = int(zf_1[n_ind1, 3])# catch up the pr_number
            pr_key = 'sl_' + str(pr_number)
            
            self.redundancy_pool[pr_key].add([nslice+1, zf_2[n_ind2, 4]])
            zf_2[n_ind2, 3] = pr_number 

#         return ind1, ind2  # return the indices which indicates the marked positions of redundancy 
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
        
        
        for nslice in np.arange(self.nz-1): 
            self._red_detect_(nslice, thresh = 1.0)
        
        # OK, let's check the the size of the pool and remove them one by one.
        dist_3d = np.zeros((0, 4)) # create an empty array to save z, y, x, f
        
         
        for sl_key, sl_value in self.redundancy_pool.items():
            z_start = sl_value.z_marker # where does the z_marker starts
            z_list = np.array(sl_value.list) # convert it into a 2d array 
            z_key = 's_' + format(z_start, '03d') 
            zframe_0 = self.z_dense[z_key]
            z_identifier = int(sl_key[3:]) - z_start*1000 # which cell? 
            
            pz = self.z_step*np.inner(z_list[:,0], z_list[:,1])/z_list[:,1].sum() # weighted average estimation 
            py, px = zframe_0[z_identifier, 0:2] # The x-y coordinates
            pf = zframe_0[z_identifier, 4] # the fluorescence 
            
            
            new_entry = np.array([[pz, py, px, pf]])
            dist_3d = np.concatenate((dist_3d, new_entry), axis = 0)
            
        
        self.dist_3d = dist_3d
           
        return dist_3d
    # ----Done with redundancy removal 
    
    

    def frame_align(self, zf_coord, z_init = 0.0, search_range = 6.0, crit = 0.80):
        """
        Purpose: align a frame's cell with 3-D reconstructed, redundancy removed distributions.
        z_frame: two-columns array containing y,x coordinates
        search_range: 
        crit: if the overall number of assigned cells exceeds crit, then the frame is aligned.   
        return: 
        """
        ncell = len(zf_coord)
#         dist_3d = self.dist_3d # catch the dist_3d
        z_dense = self.z_dense 
        
        
        n_init = int(z_init/self.z_step)
        zkey = 's_'+ format(n_init, '03d')
        z_frame = z_dense[zkey]
        
        
        