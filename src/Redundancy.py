"""
Added by Dan on 08/22/2016. To deal with redundancy. 
Can be run independently. Operated on the key-value data instead of the raw images.
After redundancy removal, the entire z-y-x-t can be saved as a 3-D npy array.
Last update: 09/12/2016
"""
import os
import numpy as np
import glob 
from numeric_funcs import circs_reconstruct
from string_funcs import path_leaf, number_strip
from Preprocess import Drift_correction

#=First of all, let's have a tiny class

class z_image():
    def __init__(self, coord, data):
        """
        This is a tiny structure for storaging the data.
        """
        self.coord = coord
        self.data = data
        self.n_cell = coord.shape[0]
        
    def coord_shift(self, dy = None, dx = None):
        """
        shift the coordinate by dy or dx
        """
        if dy is None:
            pass
        else:
            self.coord[:,0] -= dy
        
        if dx is None:
            pass
        else:
            self.coord[:,1] -= dx
        
        
# The big class for redundancy removal
        


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

            data_key = z_keys[iz]
#             data_key = format(z_keys[iz], '03d')
            z_coords[data_key] = coord
            z_fluoro[data_key] = data
            
            # add a key stripping statement here
        self.nz = nz
        self.z_keys = z_keys.sort() # sort the z_keys 
        self.z_coords = z_coords
        self.z_fluoro = z_fluoro
        self.n_blobs = n_blobs
        
    
    def zs_construct(self):
        """ 
        construct a virtual z-stack, do the alignment
        return a new z-stack 
        """
        
        temp_stack = np.zeros(self.nz, self.ny, self.nx)
        frame_size = np.array([self.ny, self.nx])
        for zz in np.arange(self.nz):
            """
            I should insert a small function to mark out all the patches on the frame first. 
            """
#             kz = self.z_keys[zz]
            data_key = self.z_keys[zz] 
            coord = self.z_coords[data_key]
            sig_aver = self.fluor[data_key].sum(axis = 0)
            nb = self.n_blobs[data_key]
            blob_list = np.zeros(nb, 3)
            blob_list[:,:2] = coord
            blob_list[:,2] = sig_aver
            
            temp_stack[zz] = circs_reconstruct(frame_size, blob_list, dr=6) # mark out very small circles to calculate correlations
        
        Z_drc = Drift_correction(temp_stack, mfit = 0)
        new_stack = Z_drc.drift_correct(offset=0, ref_first = False) # drift correct. 
        drift_list = Z_drc.drift_list
        
        for z_key, coord in self.z_coords.items():
            # correct all the cell positions with drift list 
            kz = int(z_key)
            dy = drift_list[kz,0]
            dx = drift_list[kz,1]
            
            coord[:,0] -= dy
            coord[:,1] -= dx
            
        # updated rhe coordinates of the blobs    
            
        self.z_stack = new_stack
        return new_stack


    def corrmap_stack(self):
        """
        Construct a correlation map and a distance map which can be passed to other functions. 
        """
        n_all = self.n_blobs.sum()
        self.n_cumu = self.n_blobs.cumsum()
        
        self.dist_map = np.zeros([n_all, n_all])
        self.corr_map = np.zeros([n_all, n_all])
        
        # construct the correlation map and distance map blockwise 
        for z1 in np.arange(self.nz):
            for z2 in np.arange(z1):
                self.corrmap_pair(z1, z2)
            
                
                
        



    def corrmap_pair(self, z_ref, z_corr):
        """
        detect redundancy between two adjacent slices
        Method: 
        1.Construct a distance map, calculate the distances between every two cells.
        2.Find out the correlations between every two adjacent cells, fill up the correlation map.
        """
        n_ref = self.n_blobs[z_ref]
        n_corr = self.n_blobs[z_corr]
        
        im_ref = self.z_stack[z_ref]
        im_corr = self.z_stack[z_corr]
        
        
        data_ref = self.z_fluoro[z_ref]
        data_corr = self.z_fluorop[z_corr]
        

        
        
        
        
        
        dist_map_p2 = np.zeros([n_ref, n_corr])
        corr_map_p2 = np.zeros([n_ref, n_corr])
        
        r_start = self.n_cumu[z_ref]
        r_end = self.n_cumu[z_ref+1]
        c_start = self.n_cumu[z_corr]
        c_end = self.n_cumu[z_corr+1]
        
        
        self.dist_map[r_start:r_end, c_start:c_end] = dist_map_p2
        self.corr_map[r_start:r_end, c_start:c_end] = corr_map_p2
        
        # done with updating the block, only lower left side is updated! 
        
        


    def red_detect(self, z_cross = 2):
        """
        Detect redundancy
        z_cross: up or down by z_cross number of slices. 
        """ 
        
        
        