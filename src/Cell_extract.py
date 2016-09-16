"""
Created by Dan in July-2016
Cell extraction based on the blobs_log in skimage package 
Last update: 08/18/16 
Now it really feels lousy. :( Try to use as few for loops as you can!
The class is supposed to have nothing to do with file name issue. I need to address it out of the class.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from numeric_funcs import circ_mask_patch


OL_blob = 0.8
magni_lateral = 0.295 # 0.295 micron per pixel

class Cell_extract(object):
    # this class extracts 
    def __init__(self, im_stack, diam = 6):
        self.stack = im_stack
        self.data_list = {}
        self.n_slice = im_stack.shape[0]
        self.bl_flag = np.zeros(self.n_slice).astype('int') # create an all-zero array for 
        self.frame_size = np.array(im_stack.shape[1:])
        self.blobset = [diam, diam+1, diam]
        
    def image_blobs(self, n_frame):
        """
        input: number of frame
        process: 
        0 -- recognize the blobs inside one plane (regardless of alignment)
        1 -- calculate the mean signal intensity inside the detected blobs using the 
        updated on 08/18: merge with image_signal_integ.
        
        """
        im0 = self.stack[n_frame]
        mx_sig = self.blobset[0]
        mi_sig = self.blobset[1]
        nsig = self.blobset[2]
        # comment on 09/05: we need a smarter way to do the threshold setting.
        th = (np.mean(im0)-np.min(im0))/15.0
#         print("threshold:", th)
        cblobs = blob_log(im0, 
            max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL_blob)

        n_blobs = cblobs.shape[0]
        if(n_blobs == 0):
            raise ValueError("This slice contains no blobs or has not been processed yet. ")
            self.bl_flag[n_frame] = -1
            return
        else:
            frame_size = self.frame_size
            self.bl_flag[n_frame] = n_blobs 
            data_slice = np.empty([n_blobs, 5]) # initialize an empty array
            # also, remove blobs that are on the margin
            
            for ii in np.arange(n_blobs):
                blob = cblobs[ii]
                # going through all blobs in list 
                cr = blob[0:2]
                dr = blob[-1]
#                 mask = circ_mask([self.ny, self.nx], cr, dr)
                mask = circ_mask_patch(frame_size, cr, dr) 
                signal_int = im0[mask].mean() # replace sum() with mean()
                data_slice[ii] = np.array([blob[0], blob[1], n_frame, dr, signal_int])
                
        # also, update self.data_list here instead of in stack_blobs, so you can use it right after single-frame processing!
        kwd = 's_'+ str(n_frame).zfill(3)
        self.data_list[kwd] = data_slice
        return data_slice
        # done with image_blobs
    

    def stack_blobs(self, msg = False):
        """
        process all the frames inside the stack and save the indices of frames containing blobs in self.valid_frames
        Update on 08/16: make the radius of blobs uniform.
        Update on 08/18: merge with the stack_signal_integ 
        """
        
        for n_frame in np.arange(self.n_slice):
            self.image_blobs(n_frame)
            if msg:
                n_blobs = self.bl_flag[n_frame].astype('int64')
                print("number of blobs in %d th frame: %d" %(n_frame, n_blobs))
            
        self.valid_frames = np.where(self.bl_flag>0)[0]
        # end of the function stack_blobs 
        
    
    
            # This sounds really lousy.

    def stack_signal_propagate(self, n_frame = 0):
        """
        Assume that all the slices are aligned and morphologically the same. We only extract cells 
        from one slice (usually the first one), and integrate the values at the same sites at the rest
        slices.This method should not be used in z-stacks.
        Procedures:
        0 --- calculate the blobs in the first slice
        1 --- replace the radius with the mininum radius
        2 --- assign cblobs to the clists 
        3 --- image_signal_integ
        Update on 08/19. Output: a dict. ['xy']: coordinates; ['data']: fluorescence signal.
        """
        if(np.isscalar(n_frame)):
            if self.bl_flag[n_frame]>0: # if the 
                kwd = 's_'+ str(n_frame).zfill(3)
                data_slice = self.data_list[kwd]
            else:
                data_slice = self.image_blobs(n_frame) # extract blobs from the selected frame first
            n_blobs = self.bl_flag[n_frame]
        else:
            # n_frame is a list 
            pass
        
        blob_time_stack = dict()
        coords = data_slice[:,:2] # takenout the y and x coordinates as maps
        blob_time_stack['xy'] = coords
        dr_min = np.min(data_slice[:,3])-0.5 # get an uniform dr. 
        train_signal = np.zeros((self.n_slice, n_blobs))
        
        for z_frame in np.arange(self.n_slice):
            self.bl_flag[z_frame] = n_blobs
            z_signal = self.image_signal_propagate(z_frame, coords, dr_min)
            train_signal[z_frame, :] = z_signal
            
        blob_time_stack['data'] = train_signal
        
        return blob_time_stack
    # done with stack_signal_propagate


    def image_signal_propagate(self,z_frame, maps, dr):
        """
        Added on 08/18 to replace image_signal_integ.
        The idea is similar to that in the image_blobs
        maps: the (y,x) coordinates
        return: only fluorescence instead of coordinate and fluorescence. 
        """ 
        im0 = self.stack[z_frame] 
        frame_size = self.frame_size
        nblobs = maps.shape[0]
        f_slice = np.zeros(nblobs)
        ii = 0
        for coord in maps:
            mask = circ_mask_patch(frame_size, coord, dr)
            f_slice[ii] = im0[mask].mean()
            ii += 1
        
        return f_slice 
        
        
    def save_data_list(self, dph):
        """
        Presumption: self.data_list has been fully updated 
        dph: data path + file name 
        """
        np.savez(dph, **self.data_list) # with keys saved 
    
        
        
    def stack_reload(self, new_stack):
        """
        Updates the image stack saved in the class, reset everything 
        """
        self.stack = new_stack 
        self.n_slice, ny, nx = new_stack.shape
        self.frame_size = np.array([ny,nx])
        self.bl_flag = np.zeros(self.n_slice)
        self.data_list.clear()
        print("reload completed.")
        # ----- reload the im_stack
    
    
    #----------------------- Next, let's think about data visulization -----------------------------------------
    # ------------------------This is still a dumb version.--------------------------------
    def frame_display(self, n_frame):
        """
        This function displays all the extracted cells from a selected slice. 
        """
        fig = plt.figure(figsize = (12, 5.5))
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        im0 = self.stack[n_frame]
        ax1.imshow(im0, cmap = 'Greys_r' )
        ax2.imshow(im0, cmap = 'Greys_r' )
        
        kwd = 's_'+ str(n_frame).zfill(3)
        blobs_list = self.data_list[kwd]
        for blob in blobs_list:
            y = blob[0]
            x = blob[1]
            r = blob[3]
            c = plt.Circle((x, y), 1.4*r, color='g', linewidth=1, fill=False)
            ax2.add_patch(c)
        #         print(r)
        
        plt.tight_layout() # add a tight layout 
        return fig
    

    def volume_display(self, zstep = 3.00, view_angle=None):
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
        
        