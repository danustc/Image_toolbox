"""
Added by Dan on 08/22/2016. To deal with redundancy. 
Can be run independently. Operated on the key-value data instead of the raw images.
After redundancy removal, the entire z-y-x-t can be saved as a 3-D npy array.
Last update: 09/06/2016
"""

import numpy as np
import glob 
from numeric_funcs import circ_mask_patch
from string_funcs import path_leaf, number_strip
from Preprocess import Drift_correction


class redundant(object):
    def __init__(self, work_folder, dims):
        """
        work_folder: the folder containing all the npz files.
        dims: the size of the frames, unit: pixel
        """
        self.work_folder = work_folder
        self.ny, self.nx = dims
        data_list = glob.glob(work_folder+'*.npz')

        self.nz = len(data_list)
        self.z_coords = dict()
        
        
        for dset_name in data_list:
            # load dataset one by one 
            zset = np.load(dset_name)
            coord = zset['xy']
            data = zset['data']
            z_name = path_leaf(dset_name)
            z_key = number_strip(z_name, delim = '_', ext = True)
            self.z_coords[z_key] = coord
            # add a key stripping statement here
            
    
    def __z_construct__(self):
        """ 
        construct a virtual z-stack, do the alignment 
        """
        temp_stack = np.zeros(self.nz, self.ny, self.nx)
        frame_size = np.array([self.ny, self.nx])
        for zz in np.arange(self.nz):
            # fill 
            """
            I should insert a small function to mark out all the patches on the frame first. 
            """
    
        
        Z_drc = Drift_correction(temp_stack, mfit = 0)
        new_stack = Z_drc.drift_correct(offset=0, ref_first = False)
        
        
        return new_stack


    def red_detect(self):
        """
        Detect redundancy 
        """ 
        
        
        