"""
Added by Dan on 08/22/2016. To deal with redundancy. 
Can be run independently. Operated on the key-value data instead of the raw images.
After redundancy removal, the entire z-y-x-t can be saved as a 3-D npy array.
Last update: 09/13/2016
"""
import os
import numpy as np
import glob 
from numeric_funcs import circs_reconstruct, corr_mat
from df_f import dff_raw, dff_expfilt
from string_funcs import path_leaf, number_strip
from Preprocess import Drift_correction

#=First of all, let's have a couple of small functions and a tiny class

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
# =========================================================================================================        


class Redundant_removal(object):
    def __init__(self, work_folder, dims):
        """
        work_folder: the folder containing all the npz files.
        dims: the size of the frames, unit: pixel
        temporal analysis added to convert all the raw_fluorescence into df_f.
        """
        self.work_folder = work_folder
        self.ny, self.nx = dims
        data_list = glob.glob(work_folder+'*.npz*')
        data_list.sort(key = os.path.getmtime)

        nz = len(data_list)
        z_keys = np.zeros(nz)
        n_blobs = np.zeros(nz)
        
        z_images = dict()
        
        for iz in np.arange(nz):
            # load dataset one by one
            dset_name = data_list[iz] 
            zset = np.load(dset_name)
            zm = self.dff_convert(zset, ft_width = 6)
            z_name = path_leaf(dset_name)
            z_keys[iz] = int(number_strip(z_name, delim = '_', ext = True))
            data_key = z_keys[iz]
            n_blobs[data_key] = zm.n_cell # number of blobs in each z-slice
            z_images[data_key] = zm     # data_key = format(z_keys[iz], '03d')
            
            # add a key stripping statement here
        self.nz = nz
        z_keys.sort()
        self.z_keys = z_keys # sort the z_keys 
        self.n_blobs = n_blobs
        self.z_images = z_images
        # done with z_images initialization
        
         
        
    def dff_convert(self, zset, ft_width = 6):
        """
        convert the raw data into df_f, save as z_images 
        """    
        coord = zset['xy']
        data = zset['data']
        nc = coord.shape[0]
        
        dff_r = dff_raw(data, ft_width)[0]
        dff = np.zeros_like(dff_r)
        
        for cc in np.arange(nc):
            dff[:,cc] = dff_expfilt(dff_r[:,cc], dt = 0.80, t_width = 2.0)[0]
         
        
        zm = z_image(coord, dff)
        return zm 
        
        
    
    def zs_construct(self):
        """ 
        construct a virtual z-stack, do the alignment
        return a new z-stack
        Updated on 09/13:  
        """
        
        temp_stack = np.zeros([self.nz, self.ny, self.nx])
        frame_size = np.array([self.ny, self.nx])
        for zz in np.arange(self.nz):
            """
            I should insert a small function to mark out all the patches on the frame first. 
            """
#             kz = self.z_keys[zz]
            data_key = self.z_keys[zz] 
            z_image = self.z_images[data_key]
            sig_aver = z_image.data.sum(axis = 0)
            nb = self.n_blobs[data_key]
            blob_list = np.zeros([nb, 3])
            blob_list[:,:2] = z_image.coord
            blob_list[:,2] = sig_aver
            temp_stack[zz] = circs_reconstruct(frame_size, blob_list, dr=6) # mark out very small circles to calculate correlations
        
        Z_drc = Drift_correction(temp_stack, mfit = 0)
        new_stack = Z_drc.drift_correct(offset=0, ref_first = False) # drift correct. 
        drift_list = Z_drc.drift_list
        
        for z_key in self.z_images.keys():
            # correct all the cell positions with drift list 
            kz = int(z_key)
            dy = drift_list[kz,0]
            dx = drift_list[kz,1]
            self.z_images[z_key].coord_shift(dy, dx)
            
        # updated rhe coordinates of the blobs    
            
        self.z_stack = new_stack
        return new_stack


    def corrmap_stack(self):
        """
        Construct a correlation map and a distance map which can be passed to other functions. 
        """
        n_all = self.n_blobs.sum()
        n_cumu = self.n_blobs.cumsum()
        self.n_cumu = np.concatenate(([0], n_cumu))
        
        
        self.dist_map = np.zeros([n_all, n_all])
        self.corr_map = np.zeros([n_all, n_all])
        
        # construct the correlation map and distance map blockwise 
        for z_corr in np.arange(self.nz):
            print(z_corr)
            self.corrmap_self(z_corr)
            for z_ref in np.arange(z_corr):
                self.corrmap_pair(z_ref, z_corr)
            
            
        # symmetrize self.dist_map and self.corr_map
        # WARNING!!!!! Here we cannot use "+=" operator.
        self.dist_map = self.dist_map.T + self.dist_map
        self.corr_map = self.corr_map.T + self.corr_map
        # done with corr_stack


    def corrmap_self(self, z_frame):
        """
        self correlation
        """
        
        data = self.z_images[z_frame].data
        corr_map_in = corr_mat(data)/2. # because of the symmetrization step, here the diagonal block is reduced by 2.
        
        n_sta = self.n_cumu[z_frame]
        n_end = self.n_cumu[z_frame+1]
        self.corr_map[n_sta:n_end, n_sta:n_end] = corr_map_in
        




    def corrmap_pair(self, z_ref, z_corr):
        """
        detect redundancy between two adjacent slices
        Method: 
        1.Construct a distance map, calculate the distances between every two cells.
        2.Find out the correlations between every two adjacent cells, fill up the correlation map.
        """
        im_ref = self.z_images[z_ref]
        im_corr = self.z_images[z_corr]
        
    
        # construct the dis tance map        
        coord_ref = im_ref.coord
        coord_corr = im_corr.coord
        y_ref = coord_ref[:,0]
        x_ref = coord_ref[:,1]
        y_corr = coord_corr[:,0]
        x_corr = coord_corr[:,1]
        
        YC, YR = np.meshgrid(y_corr, y_ref)
        XC, XR = np.meshgrid(x_corr, x_ref)
        dist_map_p2 = np.sqrt((YC-YR)**2 + (XC-XR)**2) # distant map
        
        # construct the correlation map        
        data_ref = im_ref.data
        data_corr = im_corr.data
        corr_map_p2 = corr_mat(data_ref, data_corr, scorr = False)
        
        
        # self.n_cumu is in the function zs_construct
        r_start = self.n_cumu[z_ref]
        r_end = self.n_cumu[z_ref+1]
        c_start = self.n_cumu[z_corr]
        c_end = self.n_cumu[z_corr+1]
        
        
        self.dist_map[r_start:r_end, c_start:c_end] = dist_map_p2
        self.corr_map[r_start:r_end, c_start:c_end] = corr_map_p2
        

        # done with updating the block, only UPPER RIGHT side is updated! 
        
    def red_detect(self, z_cross = 2):
        """
        Detect redundancy
        z_cross: up or down by z_cross number of slices. 
        """ 
        
        
        