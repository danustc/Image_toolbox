import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.feature import blob_log
from skimage.morphology import reconstruction
from common_funcs import circ_mask


OL_blob = 0.8

class Cell_extract(object):
    # this class extracts 
    def __init__(self, im_stack, diam = 5):
        self.stack = im_stack
        self.diam = diam
        self.c_list = {} # create an empty array
        self.n_slice = im_stack.shape[0]
        self.bl_flag = np.zeros(self.n_slice) # create an all-zero array for 
        self.ny, self.nx = im_stack.shape[1:]
        self.blobset = [self.diam, self.diam-1, self.diam+1]
        
    def image_blobs(self, n_frame):
        im0 = self.stack[n_frame]
        mx_sig = self.blobset[0]
        mi_sig = self.blobset[1]
        nsig = self.blobset[2]
        th = np.min(im0) # threshold
        
        self.c_list[n_frame] = blob_log(im0, 
            max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL_blob)
        self.bl_flag[n_frame] = self.c_list[n_frame].shape[0]
        # end of the function image_blobs
    

    def stack_blobs(self):
        for n_frame in np.arange(self.n_slice):
            self.image_blobs(n_frame)
            
    
    def image_signal_integ(self, n_frame):    
        """
        assume the blobs have been detected, now let's extract all the signals and save it in a huge array. 
        Column: 0 --- y coordinate, unit pixel
                1 --- x coordinate 
                2 --- z (number of slice)
                3 --- radius 
                4 --- fluorescence 
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
                
            return data_slice
            # finished of image_signal_integ
    
        
    def stack_signal_archive(self):
        """
        Once all the frames are processed for cell extraction, let's concatenate all the blob informations as a big list. 
        """
        
        
        

    

def image_dilation_trunc(im0):
    image = filters.gaussian(im0.astype('float'),1.0)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method= 'dilation')
    return dilated

    
def image_blobs(im0, blob_set = [5,4,6], th = 50, OL = 1):
    # edited on 07/25/26 
    # added the adjacent 
    
    mx_sig = blob_set[0]
    mi_sig = blob_set[1]
    nsig = blob_set[2]
    blobs_log = blob_log(im0, max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL)
    
    # let's plot the images
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(im0, cmap='Greys_r')
    ax2 = fig.add_subplot(122)
    ax2.imshow(im0, cmap='Greys_r')
    
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), np.sqrt(2)*r, color='g', linewidth=1, fill=True)
        ax2.add_patch(c)
        #         print(r)
    plt.show()
    return blobs_log # The list of traced blobs 
    
    
def stack_blobs(stack, blob_set = [5,4,6], th = 50, OL = 1):
    # extract all the blobs 
    # This is not done 
    # Updated on 07/25/2016
    blobs_stack = {} # change into a dictionary 
    for nslice in np.arange(stack.shape[0]):
        blobs_log = image_blobs(stack[nslice], blob_set, th, OL)
        blobs_stack[nslice] = blobs_log
        
        
    return blobs_stack