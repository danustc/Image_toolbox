"""
Created by Dan in July-2016
Cell extraction based on the blobs_log in skimage package 

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import blob_log
from common_funcs import circ_mask


OL_blob = 0.8
magni_lateral = 0.295 # 0.295 micron per pixel

class Cell_extract(object):
    # this class extracts 
    def __init__(self, im_stack):
        self.stack = im_stack
        self.c_list = {} # create an empty array
        self.data_list = {}
        self.n_slice = im_stack.shape[0]
        self.bl_flag = np.zeros(self.n_slice) # create an all-zero array for 
        self.ny, self.nx = im_stack.shape[1:]
       
        
    def image_blobs(self, n_frame):
        """
        input: number of frame
        """
        im0 = self.stack[n_frame]
        mx_sig = self.blobset[0]
        mi_sig = self.blobset[1]
        nsig = self.blobset[2]
#         th = (np.max(im0)-np.min(im0))/10. # threshold
        th = (np.mean(im0)-np.min(im0))/12.
        print("threshold:", th)
        self.c_list[n_frame] = blob_log(im0, 
            max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL_blob)
        self.bl_flag[n_frame] = self.c_list[n_frame].shape[0]
        # end of the function image_blobs
    

    def stack_blobs(self, diam = 6):
        """
        process all the frames inside the stack and save the indices of frames containing blobs in self.valid_frames
        """
        self.diam = diam
        self.blobset = [self.diam-1, self.diam+1, self.diam]
            
        for n_frame in np.arange(self.n_slice):
            self.image_blobs(n_frame)
            
        self.valid_frames = np.where(self.bl_flag>0)[0]
        # end of the function stack_blobs 
        
    
    def redundant_removal(self, n_spread = 3, p_thresh = 2 ):
        """
        remove the redundancy
        idea: sort the blobs list according to the x,y position
        Count every n_spread adjacent slices to remove the redundant ones 
        criteria: z continued, center difference < p_thresh pixel
        
        """
        # Qia tou qu wei! 
        for n_frame in np.arange(1, self.n_slice-1):
            self.data_list[n_frame]
            
        
        
        
        
    
    def image_signal_integ(self, n_frame):    
        """
        assume the blobs have been detected, now let's extract all the signals and save it in a huge array. 
        Column: 0 --- y coordinate, unit pixel
                1 --- x coordinate 
                2 --- z (number of slice)
                3 --- radius 
                4 --- fluorescence
        Update self.data_list 
        """
        im0 = self.stack[n_frame]
        blbs = self.c_list[n_frame]
        n_blobs = self.bl_flag[n_frame] # number of blobs in each slice
        if(n_blobs == 0):
            raise ValueError("This slice contains no blobs or has not been processed yet. ")
        
        else: 
            data_slice = np.empty([n_blobs, 5]) # initialize an empty array
            for ii in np.arange(n_blobs):
                blob = blbs[ii]
                # going through all blobs in list 
                cr = blob[0:2]
                dr = blob[-1]
                mask = circ_mask([self.ny, self.nx], cr, dr)
                signal_int = im0[mask].sum()
                data_slice[ii] = np.array([blob[0], blob[1], n_frame, dr, signal_int])
                self.data_list[ii] = data_slice
            return data_slice
            # finished of image_signal_integ
    
        
    def stack_signal_archive(self):
        """
        Once all the frames are processed for cell extraction, let's concatenate all the blob informations as a big list. 
        """
        # Let's find out all the slices that are processed.
        valid_frames = self.valid_frames 
        n_total = self.bl_flag[valid_frames].sum() # The total number of blobs
        blobs_archive = np.empty([n_total, 5])

        # archiving the blobs list         
        n_sta = 0 
        for n_frame in valid_frames:
            n_blobs = self.bl_flag[n_frame].astype('int64')
            self.data_list[n_frame] = self.image_signal_integ(n_frame)
            blobs_archive[n_sta:n_sta+n_blobs] = self.data_list[n_frame]
            print(self.data_list[n_frame])
            
            n_sta +=n_blobs
        
        # should I save the blobs somewhere automatically?
        # after archiving blobs list, clear the dictionary
        self.blobs_archive = blobs_archive
        self.data_list.clear() 
        return blobs_archive    
        # finished of stack_signal_archive 
        
        
    def save_archive(self, dph):
        """
        dph: data path + file name 
        """
        np.save(dph, self.blobs_archive)
        self.blobs_archive = []
        print("The archive has been saved and the array cleaned.")
        
        
    def stack_reload(self, new_stack):
        """
        Updates the image stack saved in the class, reset everything 
        """
        self.stack = new_stack 
        self.n_slice, self.ny, self.nx = new_stack.shape
        self.bl_flag = np.zeros(self.n_slice)
        self.c_list.clear()
        self.data_list.clear()
        print("reload completed.")
        # ----- reload the im_stack
        
    #----------------------- Next, let's think about data visulization -----------------------------------------
    
    def frame_display(self, n_frame, pxl_cvt = False):
        """
        This function displays all the extracted cells from a selected slice. 
        """
        fig = plt.figure(figsize = (12, 5.5))
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        im0 = self.stack[n_frame]
        ax1.imshow(im0, cmap = 'Greys_r' )
        ax2.imshow(im0, cmap = 'Greys_r' )
        
        blobs_list = self.c_list[n_frame]
        for blob in blobs_list:
            y, x, r = blob
            c = plt.Circle((x, y), np.sqrt(2)*r, color='g', linewidth=1, fill=False)
            ax2.add_patch(c)
        #         print(r)
        return fig
    

    def volume_display(self, zstep = 3.00):
        """
        This is a daring attempt for 3-D plots of all the blobs in the brain. 
        Prerequisites: self.blobs_archive are established.
        The magnification of the microscope must be known.
        """
        fig3 = plt.figure(figsize = (12, 9))
        ax_3d = fig3.add_subplot(111, projection = '3d')
        
        for n_frame in self.valid_frames:
            # below the coordinates of the cells are computed.
            blobs_list = self.c_list[n_frame] 
            ys = blobs_list[:,0]*magni_lateral
            xs = blobs_list[:,1]*magni_lateral 
            zs = np.ones(len(blobs_list))*n_frame*zstep
            ss = (np.sqrt(2)*blobs_list[:,2]*magni_lateral)**2
            ax_3d.scatter(xs,ys, zs, zdir = 'z', s=ss, c='g')        
        
        
        
        ax_3d.set_xlabel('x (micron)', fontsize = 12)
        ax_3d.set_ylabel('y (micron)', fontsize = 12)
        ax_3d.set_zlabel('z (micron)', fontsize = 12)
        return fig3
        