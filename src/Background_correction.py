"""
# ----------- Author: Dan Xie ------------------------
# ------------Last update: 09/16/2016 --------------------------

for pre-processing (actually pre-preprocessing.)
This file has two classes: 

1. The class Deblur deblurrs the in-plane image and deblurrs each image from its adjacent stacks 
2. The class Drift_correction corrects the drifts between adjacent slices based on cross-correlation function.
To be updated: replace all the global variables with local variables.


"""
import tifffunc
import numpy as np
from skimage import filters
from numeric_funcs import histo_peak
import ntpath

# --------------------------------------Functions ----------------------------------------



def crop_stack(dph, c_nw, c_se, post_fix = 'cr'):
    """
    c_nw: northwest corner 
    c_se: southeast corner 
    dph: the raw stack path, contains the absolute path and the filename.
    """
    stack = tifffunc.read_tiff(dph)
    head, tail = ntpath.split(dph)
    cr_name  = tail[:-4]
    
    if(cr_name[-1].isdigit()):
        ii = -2
        while(cr_name[ii].isdigit()):
            ii-=1
        cr_name_h = cr_name[:(ii+1)]
        cr_name_t = cr_name[(ii+1):]
        fname = ''.join([head, cr_name_h, post_fix, cr_name_t, tail[-4:]])
        
    else:
        fname = ''.join([head, cr_name, post_fix, tail[-4:]])
    
    y1, x1 = c_nw
    y2, x2 = c_se 

    try:
        cr_stack = stack[:,y1:y2, x1:x2]
    except IndexError:
        cr_stack = 'null'
        
    tifffunc.write_tiff(cr_stack, fname)
    return cr_stack
        
# ----------------------------Done with crop    
    

# --------------------------------------Below are classes for deblur and drift correction 

class Deblur(object):
    def __init__(self, stack, sig = 30):
        """
        Update on 0
        sig is the width of Gaussian filter
        """
        self.stack = np.copy(stack).astype('float64')
        self.current = 0 # just using an index
        self.Nslice = self.stack.shape[0] # number of slices 
        print("The number of slices:", self.Nslice)
        self.px_num = self.stack.shape[1:] # the number of pixels in y and x for each slice
        print("The shape of images:", self.px_num)
        self.status = -1* np.ones(self.Nslice) # the status 
        self.sig = sig
        
        """
        status = -1: the slices are raw
        status = 0: current slice deblurred and aligned 
        status = 1: current slice deblurred but not aligned 
        """
    
    
    def image_high_trunc_inplane(self, nslice = None, ofst = 1.00):
        """
        # the idea comes from Nature-scientific reports,  3:2266.
        # im0: image 0 
        # method: filter method
        # ext: extension of the filter 
        # sig: width of the Gaussian filter
        """
        sig = self.sig
        if (nslice is None):
            im0 = self.stack[self.current]
        else:
            im0 = self.stack[nslice]
            
        im0[im0 ==0] = ofst # remove all the zero pixels 
            
        ifilt = filters.gaussian(im0, sigma=sig)
        iratio = im0/ifilt
        nmin = np.argmin(iratio) 
        gmin_ind = np.unravel_index(nmin, im0.shape) # global mininum of the index    
        sca =im0[gmin_ind]/(ifilt[gmin_ind])
#         print("scale:",sca)
        im0 -= (ifilt*sca*0.999) # update the background-corrected image
#         return im0, ifilt



    def image_high_trunc_adjacent(self, wt = 0.40, sca = 3.0):
        sig = sca*self.sig # the Gaussian filter width of the adjacent slice 
        # Should this be done after the in-plane deblurring is accomplished, or before that?
        n_current = self.current 
        if n_current > 0 and n_current < (self.Nslice-1):
            i_mid = self.stack[n_current]
            i_up = self.stack[n_current-1] # the previous slice 
            i_down = self.stack[n_current+1] # the next slice 
            ifilt_up = filters.gaussian(i_up, sigma = sig)
            ifilt_down = filters.gaussian(i_down, sigma = sig)
            
            up_ratio = i_mid/ifilt_up
            down_ratio = i_mid/ifilt_down
            nmin_up = np.argmin(up_ratio)
            nmin_down = np.argmin(down_ratio)
            
            gind_up = np.unravel_index(nmin_up, self.px_num)
            gind_down = np.unravel_index(nmin_down, self.px_num)
            print(gind_up, gind_down)
            sca_up = i_mid[gind_up]/ifilt_up[gind_up]
            sca_down = i_mid[gind_down]/ifilt_down[gind_down]
            
            print('%.6f' %sca_up, '%.6f' %sca_down)
            
            i_mid -= wt*(sca_up*ifilt_up+sca_down*ifilt_down) # Pay attention: it means self.stack changes!
            return i_mid
        else: 
            pass    
            # only the slices not on the border are corrected  
        
        # truncate image from neighbors

    def stack_high_trunc(self, adjacent = False, wt = 0.40):
    # run stack_high_trunc for a whole stack
    # updated: make adjacent optional
        for ii in np.arange(self.Nslice):
            self.current = ii
            if(adjacent):
                self.image_high_trunc_adjacent(wt) # subtract the adjacent plane values first 
            self.image_high_trunc_inplane()  # in-plane correction


    def baseline_redef(self):
        """
        Added on 09/13, aims at removing the artefacts from background subtraction.
        must be performed after in-plane background subtraction. 
        """
        ny, nx = self.px_num
        nbin = int(2*np.log2(ny*nx)) # number of bins
        
        px_max = np.zeros(self.Nslice)
        
        for ns in np.arange(self.Nslice):
            im_bs = self.stack[ns]
            val_cut = np.mean(im_bs)*0.5 # where to cut off 
            px_max[ns] = histo_peak(im_bs, val_cut, nbin)
        
            
        self.px_max = px_max 
        return px_max


    def get_stack(self):
        """
        simply return the stack
        """
        return self.stack

    
    def write_stack(self, n_apdx):
        """
        Update on 08/15: 
        the tifffile module does not allow overwriting so we are going to write it ourselves
        
        """
        fext = self.impath[:-4]
        tifffunc.write_tiff(self.stack, fext+n_apdx+'.tif')
    
        

