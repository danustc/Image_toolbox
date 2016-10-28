"""
Created on 09/13/2016 by Dan. 
The purpose is to postprocess the extracted fluorescence signals into df_f (raw and filtered)
"""

import os 
import glob
import df_f 
import numeric_funcs as numfc
from string_funcs import path_leaf, number_strip
import numpy as np
from Alignments import Drift_correction, cross_alignment
# -----------------------------First, let's define a small class storing everything 

class z_image():
    def __init__(self, coord, f_raw):
        """
        This is a tiny structure for storaging the data.
        """
        self.coord = coord
        self.f_raw = f_raw
        self.n_time, self.n_cell = f_raw.shape
    
            
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

    
    
    def dff_calc(self, wd = 6):
        """
        convert the raw_fluorescence into dff
        """
        dff_r = df_f.dff_raw(self.f_raw, ft_width = wd)[0] # the return has two objects, and we only take the first one.
        dff = np.zeros_like(dff_r)
        for cc in np.arange(self.n_cell):
            dff[:,cc] =df_f.dff_expfilt(dff_r[:,cc], dt = 0.80, t_width = 2.0)[0]
        
        self.dff = dff 
        return dff
    
    
    def bright_histo(self):
        """
        construct a brightness histogram for each time-point 
        """
        nbin = int(self.n_cell*0.20)
        print(nbin)
        histo_bg = np.zeros(self.n_time)
        
        for nt in np.arange(self.n_time):
            fr = self.f_raw[nt] # take the nt-th row (not column!!!) within the same time point
            val_cut = np.mean(fr)*0.50
            pv, hist  = numfc.histo_peak(fr, val_cut, nbin)
            histo_bg[nt] = pv
            print(pv)
#             self.f_raw[nt] -= pv # correct by offset
        
#         self.f_raw -= np.min(self.f_raw) # let's make it positive.
        
        return histo_bg, hist
        
"""
Done with z_image. Next, let's define another class to construct the distance and correlation map.
"""

class img_volume(object):
    """
    a processed image volume with all the feature of interests arranged in coordinates and signals
    """
    def __init__(self, c_coords, c_signals):
        """
        zd_stack: 
        """
        self.c_coords = c_coords
        self.c_signals = c_signals
    
    # ---------------------------------------------------------------------
    
    
    

class over_counting(object):
    """
    A tiny tiny class for conveniently registering redundant cells.
    """
    def __init__(self, cell_mark, sig_1, sig_2):
        
        self.cm = cell_mark
        self.n_fold = 2
        self.signal = np.array([sig_1, sig_2])
    
        
    def add_one(self, sig_add):
        """
        If redundancy is detected in a new frame, add signal by 1 and append the signal list by sig_add.
        """
        self.n_fold +=1
        self.signal = np.concatenate(self.signal, [sig_add]) 

        
    def zdef_fitting(self):
        """
        define the z-position by gaussian fitting. Only works when self.n_fold >=3.
        """
        if self.n_fold < 3:
            pass
        
    
    
    




#================================================================================================================
class brain_construct(object):
    """
    This class contains all the data structures that suitable for future analysis, including:
    1. a list of z_image classes
    2. a reference blobs positions for registration
    3. a distance map between every two cells
    4. a correlation map between every two cells.
    """
        
    def __init__(self, work_folder, dims):
        
        self.work_folder = work_folder
        self.ny, self.nx = dims
        data_list = glob.glob(work_folder+'*.npz*')
        data_list.sort(key = os.path.getmtime)

        nz = len(data_list)
        print(nz)
        z_keys = np.zeros(nz)
        n_blobs = np.zeros(nz)
        
        z_images = dict()
        
        for iz in np.arange(nz):
            # load dataset one by one
            dset_name = data_list[iz] 
            zset = np.load(dset_name)
            coord = zset['xy']
            f_raw = zset['data']
            zm = z_image(coord, f_raw)
            z_name = path_leaf(dset_name)
            z_keys[iz] = number_strip(z_name, delim = '_', ext = True)
            data_key = z_keys[iz]
            n_blobs[data_key] = zm.n_cell # number of blobs in each z-slice
            z_images[data_key] = zm     # data_key = format(z_keys[iz], '03d')
            
            # add a key stripping statement here
        self.nz = nz
        z_keys.sort()
        self.z_keys = z_keys # sort the z_keys 
        self.n_blobs = n_blobs
        self.z_images = z_images
        
        
        
    def zs_construct(self):
        """ 
        construct a virtual z-stack, do the alignment
        return a new z-stack
        Updated on 09/13:  
        """
#         self.histo_bg = dict()
        
        
        
        temp_stack = np.zeros([self.nz, self.ny, self.nx])
        frame_size = np.array([self.ny, self.nx])
        for zz in np.arange(self.nz):
            """
            I should insert a small function to mark out all the patches on the frame first. 
            """
#             kz = self.z_keys[zz]
            data_key = self.z_keys[zz] 
            z_image = self.z_images[data_key]
            # // z_image bright_histo
            
            sig_aver = z_image.f_raw.sum(axis = 0)
            nb = z_image.n_cell
            blob_list = np.zeros([nb, 3])
            blob_list[:,:2] = z_image.coord
            blob_list[:,2] = sig_aver
            temp_stack[zz] = numfc.circs_reconstruct(frame_size, blob_list, dr=6) # mark out very small circles to calculate correlations
        
        Z_drc = Drift_correction(temp_stack, mfit = 0)
        Z_drc.drift_correct(offset=0, ref_first = False, roll_back= True) # drift correct. 
        drift_list = Z_drc.drift_list
        new_stack = Z_drc.get_stack()
        
        for z_key in self.z_images.keys():
            # correct all the cell positions with drift list
            # also, calculate the dff 
            kz = int(z_key)
            dy = drift_list[kz,0]
            dx = drift_list[kz,1]
            self.z_images[z_key].coord_shift(dy, dx)
            self.z_images[z_key].dff_calc()
#             self.histo_bg[z_key] = z_image.bright_histo()
        # updated rhe coordinates of the blobs    
            
        self.z_stack = new_stack
        return new_stack
    
    
    
    def corrmap_stack(self):
        """
        Construct a correlation map and a distance map which can be passed to other functions. 
        """
        n_all = self.n_blobs.sum()
        n_cumu = self.n_blobs.cumsum()
        n_cumu = np.concatenate(([0], n_cumu))
        self.n_cumu = n_cumu
        
        self.dist_map = np.zeros([n_all, n_all])
        self.corr_map = np.zeros([n_all, n_all])
        
        # construct the correlation map and distance map blockwise 
        for z_corr in np.arange(self.nz):
            print(z_corr)
            c_start = n_cumu[z_corr]
            c_end = n_cumu[z_corr+1]
            self._corrmap_self_(z_corr)
            for z_ref in np.arange(z_corr):
                dist_map_p2, corr_map_p2 = self._corrmap_pair_(z_ref, z_corr)
                r_start = n_cumu[z_ref]
                r_end = n_cumu[z_ref+1]
                
                self.dist_map[r_start:r_end, c_start:c_end] = dist_map_p2
                self.corr_map[r_start:r_end, c_start:c_end] = corr_map_p2
            
        # symmetrize self.dist_map and self.corr_map
        # WARNING!!!!! Here we cannot use "+=" operator.
        self.dist_map = self.dist_map.T + self.dist_map
        self.corr_map = self.corr_map.T + self.corr_map
        # done with corr_stack


    def _corrmap_self_(self, z_frame):
        """
        self correlation, private function.
        """
        
        data = self.z_images[z_frame].dff
        corr_map_in = numfc.corr_mat(data)*0.5 # because of the symmetrization step, here the diagonal block is reduced by 2.
        
        coord = self.z_images[z_frame].coord
        
        coord_y = coord[:,0]
        coord_x = coord[:,1]
        
        YC, YR = np.meshgrid(coord_y, coord_y)
        XC, XR = np.meshgrid(coord_x, coord_x)
        
        dist_map_in = np.sqrt((YC-YR)**2 + (XC-XR)**2)*0.5
        
        
        n_sta = self.n_cumu[z_frame]
        n_end = self.n_cumu[z_frame+1]
        self.corr_map[n_sta:n_end, n_sta:n_end] = corr_map_in
        self.dist_map[n_sta:n_end, n_sta:n_end] = dist_map_in
        # done with _corrmap_self_



    def _corrmap_pair_(self, z_ref, z_corr):
        """
        detect redundancy between two adjacent slices, private function.
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
        data_ref = im_ref.dff
        data_corr = im_corr.dff
        corr_map_p2 = numfc.corr_mat(data_ref, data_corr, scorr = False)
        
        
        # self.n_cumu is in the function zs_construct
        
        
        
        return dist_map_p2, corr_map_p2
        # done with _corrmap_pair_
        
        
    def red_detect(self, d_crt = 3.0, c_thresh = 0.7):
        """
        detect redundancy by 3 conditions:
        0. the center of the two cells are smaller than 
        1. distance between the two cells are less than d_crt
        2. correlation is larger than the c_thresh  
        """
        
        n_cumu = self.n_cumu
        red_list = dict()
        
        
        for na in np.arange(self.nz-1):
            n_start = n_cumu[na]
            n_mid = n_cumu[na+1]
            n_end = n_cumu[na+2]
            
            dist_map_nb = self.dist_map[n_start:n_mid, n_mid:n_end]
            corr_map_nb = self.corr_map[n_start:n_mid, n_mid:n_end]
            
            red_pos = np.logical_and(dist_map_nb < d_crt, corr_map_nb > c_thresh)
            red_indy, red_indx = np.where(red_pos) # the indices of redundancy
            zkey = format(na, '03d') 
            red_list[zkey] = (red_indy, red_indx)
            print("redundancy:", red_pos.sum())
            
            
#             dist_map_nb[red_pos] = 0. # mark out dist_map_nb
#             corr_map_nb[red_pos] = 0. # mark out corr_map nb 
            
        return red_list
        # done with red_detect
        
    
    
    
    
    def dense_regist(self, z_dense_stack):
        """
        load a densely imaged stack as a reference. (Key values only) 
    
        But how to utilize this part of information?
        Assumption: the z_dense stack has the same start and end point as the z_sparse stack does.
        """
        
        self.z_dense = z_dense_stack
        self.nz, self.ny, self.nx = z_dense_stack.shape
        
    
    
        
    def zstack_shiftcalc(self, zd_path, meanframes = 3, sample_strategy = 'uni'):
        """
        load a known, prealigned Z-stack, calculate the shifts of each time stacks with respect to the z-stack.
        meanframes: number of frames that should be taken for averaged alignment
        sample_strategy: uni --- uniformly sample among the t-stack (default)
                         first --- take the first <meanframes> frames for drift calculation
                         last --- take the last <meanframes>
        """
        zp_drift = np.zeros([self.n_ZP, 2])
        zd_ref = np.load(zd_path)
        
       
