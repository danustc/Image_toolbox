"""
Added by Dan on 08/22/2016. To deal with redundancy. 
Can be run independently. Operated on the key-value data instead of the raw images.
After redundancy removal, the entire z-y-x-t can be saved as a 3-D npy array.
Last update: 09/09/2016
"""
import os
import numpy as np
import glob 
from numeric_funcs import circs_reconstruct
from string_funcs import path_leaf, number_strip
from Preprocess import Drift_correction


class Redundant_removal(object):
    def __init__(self, work_folder, dims):
        """
        work_folder: the folder containing all the npz files.
        dims: the size of the frames, unit: pixel
        """
        self.work_folder = work_folder
        self.ny, self.nx = dims
        data_list = glob.glob(work_folder+'*.npz')
        data_list.sort(key = os.path.getmtime)

        nz = len(data_list)
        z_keys = np.zeros(nz)
        n_blobs = np.zeros(nz)
        
        z_coords = dict()
        z_fluoro = dict()
        
        for iz in np.arange(nz):
            # load dataset one by one
            dset_name = data_list[nz] 
            zset = np.load(dset_name)
            coord = zset['xy']
            data = zset['data']
            z_name = path_leaf(dset_name)
            z_keys[iz] = number_strip(z_name, delim = '_', ext = True)
            n_blobs[z_keys[iz]] = coord.shape[0] # number of blobs in each z-slice

            data_key = format(z_keys[iz], '03d')
            z_coords[data_key] = coord
            z_fluoro[data_key] = data
            
            # add a key stripping statement here
        self.nz = nz
        self.z_keys = z_keys.sort() # sort the z_keys 
        self.z_coords = z_coords
        self.z_fluoro = z_fluoro
        self.n_blobs = n_blobs
        
    
    def __z_construct__(self):
        """ 
        construct a virtual z-stack, do the alignment
        construct a virtual  
        """
        
        temp_stack = np.zeros(self.nz, self.ny, self.nx)
        frame_size = np.array([self.ny, self.nx])
        for zz in np.arange(self.nz):
            """
            I should insert a small function to mark out all the patches on the frame first. 
            """
            kz = self.z_keys[zz]
            data_key = format(kz, '03d') 
            
            coord = self.z_coords[data_key]
            sig_aver = self.fluor[data_key].sum(axis = 0)
            nb = self.n_blobs[kz]
            
            blob_list = np.hstack((coord, sig_aver)) 
            
            temp_stack[zz] = circs_reconstruct(frame_size, blob_list)
        
        Z_drc = Drift_correction(temp_stack, mfit = 0)
        new_stack = Z_drc.drift_correct(offset=0, ref_first = False) # drift correct. 
        
        
        return new_stack


    def red_detect(self, z_cross = 2):
        """
        Detect redundancy
        z_cross: up or down by z_cross number of slices. 
        """ 
        
        
        