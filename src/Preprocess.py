"""
# ----------- Author: Dan Xie ------------------------
# ------------Last update: 08/15/2016 --------------------------

for pre-processing (actually pre-preprocessing.)
This file has two classes: 

1. The class Deblur deblurrs the in-plane image and deblurrs each image from its adjacent stacks 
2. The class Drift_correction corrects the drifts between adjacent slices based on cross-correlation function.
To be updated: replace all the global variables with local variables.


"""
import tifffunc
import numpy as np
from skimage import filters
from scipy.ndimage.filters import uniform_filter
import scipy.fftpack as fftp
from numeric_funcs import fitgaussian2D
import ntpath

# --------------------------------------Functions ----------------------------------------

def res_ind(indi, hdim):
#   hdim: the dimension of the array. If the indi is larger than half of hdim, then indi is replaced by indi-hdim.
    if(indi>hdim/2):
        indo = indi - hdim
    else:
        indo = indi
    return indo




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
    def __init__(self, impath, sig = 30, ftype = 'g'):
        """
        Update on 0
        sig is the width of Gaussian filter
        """
        self.raw_stack = tifffunc.read_tiff(impath).astype('float')# load a raw image and convert to float type
        self.new_stack = np.copy(self.raw_stack)
        self.impath = impath
        self.cell_list = []
        self.current = 0 # just using an index
        self.Nslice = self.raw_stack.shape[0] # number of slices 
        print("The number of slices:", self.Nslice)
        self.px_num = self.raw_stack.shape[1:] # the number of pixels in y and x for each slice
        print("The shape of images:", self.px_num)
        self.status = -1* np.ones(self.Nslice) # the status 
        self.sig = sig
        self.ftype = ftype # specify the filter type
        
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
            im0 = self.new_stack[self.current]
        else:
            im0 = self.new_stack[nslice]
            
        im0[im0 ==0] = ofst # remove all the zero pixels 
            
        if(self.ftype == 'g'):
            ifilt = filters.gaussian(im0, sigma=sig)
        else:
            ifilt = uniform_filter(im0, size = sig) # test uniform filter
        iratio = im0/ifilt
        nmin = np.argmin(iratio) 
        gmin_ind = np.unravel_index(nmin, im0.shape) # global mininum of the index    
        sca =im0[gmin_ind]/(ifilt[gmin_ind])
#         print("scale:",sca)
        im0 -= (ifilt*sca*0.999) # update the background-corrected image
        return im0, ifilt



    def image_high_trunc_adjacent(self, wt = 0.40, sca = 3.0):
        sig = sca*self.sig # the Gaussian filter width of the adjacent slice 
        # Should this be done after the in-plane deblurring is accomplished, or before that?
        n_current = self.current 
        if n_current > 0 and n_current < (self.Nslice-1):
            i_mid = self.raw_stack[n_current]
            i_up = self.raw_stack[n_current-1] # the previous slice 
            i_down = self.raw_stack[n_current+1] # the next slice 
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
            
            i_mid -= wt*(sca_up*ifilt_up+sca_down*ifilt_down) # Pay attention: it means self.raw_stack changes!
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

        return self.new_stack
        # done with stack_high_trunc

    
    def write_stack(self, n_apdx):
        """
        Update on 08/15: 
        the tifffile module does not allow overwriting so we are going to write it ourselves
        
        """
        fext = self.impath[:-4]
        tifffunc.write_tiff(self.new_stack, fext+n_apdx+'.tif')
    
        

class Drift_correction(object):
    '''
    # updated on 08/30.  
    mfit: 0 --- linear correction
          1 --- Gaussian correction 
          2 --- nonlinear correction?  
    '''
    def __init__(self, raw_stack, mfit = 0):
        """
        do not save the raw_stack in the class.
        """
        self.stack = raw_stack
        self.nslices = raw_stack.shape[0] # number of slices
        self.idim = np.array(raw_stack.shape[1:])
        self.mfit = mfit 
        self.drift = np.zeros([self.nslices,2])
        # have a raw stack
    
    def _shift_calculation(self, ft_ref,ft_cor):
        """
        only ft_frames are passed into this function, but the fr_frames can be reused outside. 
        """
        
        F_prod = np.conj(ft_ref)*ft_cor
        X_corr = np.abs(fftp.ifft2(F_prod))

              
        # findout the maximum of X_corr and fit to the Gaussian 
        Xmax = np.argmax(X_corr)
        nrow, ncol = np.unravel_index(Xmax, X_corr.shape)
        
        dry=res_ind(nrow,self.idim[0])
        drx=res_ind(ncol,self.idim[1])
        
        
        if(self.mfit == 0):
            drift = [dry, drx]
#             print("Drifted pixel:", drift)
            
        else: # mfit is the width of gaussian function
            # This should be tested. 
            ncompy = np.round(self.idim[0]/4).astype('int64')
            ncompx = np.round(self.idim[1]/4).astype('int64')
            
            # x part: 
            if(np.abs(drx)<ncompx):
                X_corr = np.roll(X_corr,ncompx*2, axis = 1)
                Cnx = drx + ncompx*2
            else:
                Cnx = np.mod(drx, self.idim[1])  
               
            # y part:    
                
            if(np.abs(dry)<ncompy):
                X_corr = np.roll(X_corr, ncompy*2, axis = 0)    
                Cny = dry + ncompy*2
            else:
                Cny = np.mod(dry, self.idim[0])

            self.xcorr = X_corr
            xrange = np.arange(Cnx-self.mfit, Cnx+ self.mfit+1) # Again, this "+1" is due to the subtle differences between matlab and python. 
            yrange = np.arange(Cny-self.mfit, Cny+ self.mfit+1)                
            corr_profile = X_corr[yrange,:][:,xrange]
            # next, let's determine the initial value of the arguments 
            
            popt = fitgaussian2D(corr_profile)
            cx = popt[1]
            cy = popt[2]
#             dx = popt[3]
#             dy = popt[4]
#             print(cy,cx, dx, dy)
            
#             print("center found at:", cx, cy)
            drift = [dry,  drx] + np.round([cy, cx]).astype('int64')-self.mfit
#         print(dry, drx)
        print("drift:", drift)            
        return drift
            
    
    def drift_correct(self, offset = 1, ref_first = False):
        """
        Update on 09/09: keep the drift list, so the cell coordinates might need be updated.
        """
        # offset = m: start from the mth slice  
#         self.mfit = mfit
        iref = offset
        im_ref = self.stack[iref]
        ft_ref = fftp.fft2(im_ref)
        drift_list = np.zeros(self.nslices, 2)
        
        if (ref_first == False):
            for icor in np.arange(offset+1,self.nslices):
                im_cor = self.stack[icor]
                ft_cor = fftp.fft2(im_cor)
                drift = self._shift_calculation(ft_ref, ft_cor)
                im_cor = np.roll(im_cor, -drift[0], axis = 0)
                im_cor = np.roll(im_cor, -drift[1], axis = 1)
                self.stack[icor] = im_cor
                drift_list[icor] = drift
                ft_ref = ft_cor # reuse ft_ref
        else:
            # take the first slice as the reference
            for icor in np.arange(offset+1,self.nslices):
                im_cor = self.stack[icor]
                ft_cor = fftp.fft2(im_cor)
                drift = self._shift_calculation(ft_ref, ft_cor)
#                 print(drift)
                im_cor = np.roll(im_cor, -drift[0], axis = 0)
                im_cor = np.roll(im_cor, -drift[1], axis = 1)
                self.stack[icor] = im_cor
                drift_list[icor] = drift
                # differs from the ref_first False case by the last statement
        
        self.drift_list = drift_list
        
        return self.stack
    


#-----------------------------------------------------------------------------------------
