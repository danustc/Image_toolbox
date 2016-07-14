import numpy as np
import tifffunc
import glob
import matplotlib.pyplot as plt
import trackpy as tp
from skimage import filters
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.transform import hough_ellipse
from skimage import img_as_float
from skimage.morphology import reconstruction

# class Cell_exact:
class IM_stack(object):
    def __init__(self, impath):
        self.raw_stack = tifffunc.read_tiff(impath) # load a raw image 
        self.cell_list = []
        
        self.current_image = []
        
        
    def image_high_trunc(self, sig = 25):
    # the idea comes from Nature-scientific reports,  3:2266.
    # im0: image 0 
    # method: filter method
    # ext: extension of the filter 
    # sig: width of the Gaussian filter
        im0 = self.current_image
        ifilt = filters.gaussian(im0.astype('float'), sigma=sig)
        iratio = im0.astype('float')/ifilt
        nmin = np.argmin(iratio) 
        gmin_ind = np.unravel_index(nmin, im0.shape) # global mininum of the index    
        sca =im0[gmin_ind]/(ifilt[gmin_ind])
        print(sca)
        isub = im0.astype('float')-ifilt*sca # update the background-corrected image
        isub = isub.astype('uint16')
        return isub



    def image_histo(self,im0, nbins = 100):
        hist, edge_bins = np.histogram(im0, nbins)
        nbp = np.argmax(hist) # the bin location of peak 
        thrd = (edge_bins[nbp] + edge_bins[nbp+1]) / 2.
        return thrd, hist  # return: threshold 
        

    def stack_high_trunc(self, stack, sig=25):
    # run stack_high_trunc for a whole stack 
        new_stack = np.zeros_like(stack, dtype = 'uint16')
        ii = 0
        for sli in stack:
            sub = self.image_high_trunc(sli, sig)
            new_stack[ii] = sub
            ii+=1
        return new_stack




def image_dilation_trunc(im0):
    image = filters.gaussian(im0.astype('float'),1.0)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method= 'dilation')
    return dilated

    
def image_blobs(im0, blob_set = [5,4,6], th = 50, OL = 1):
    mx_sig = blob_set[0]
    mi_sig = blob_set[1]
    nsig = blob_set[2]
    blobs_log = blob_log(im0, max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL)
    fig=plt.subplot()
    ax = fig.axes
    ax.imshow(im0, cmap='Greys_r')
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), np.sqrt(2)*r, color='g', linewidth=1, fill=True)
        ax.add_patch(c)
        #         print(r)
    plt.show()
    return blobs_log # The list of traced blobs 
    
    
def stack_blobs(stack, blob_set = [5,4,6], th = 50, OL = 1):
    # extract all the blobs 
    
    blobs_stack = []
    for img in stack:
        blobs_log = image_blobs(img, blob_set, th, OL)
        blobs_stack = np.concatenate()
        
        
        