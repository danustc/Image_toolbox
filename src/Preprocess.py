"""
# ----------- Author: Dan Xie ------------------------
# ------------Last update: 08/03/2016 --------------------------
This file has two classes: 

1. The class Deblur deblurrs the in-plane image and deblurrs each image from its adjacent stacks 
2. The class Drift_correction corrects the drifts between adjacent slices based on cross-correlation function.

"""


import tifffunc
import numpy as np
from skimage import filters
import scipy.fftpack as fftp
from common_funcs import fitgaussian2D


class Deblur(object):
    def __init__(self, impath, sig = 30):
        # sig is the width of Gaussian filter 
        self.raw_stack = tifffunc.read_tiff(impath).astype('float')# load a raw image and convert to float type
        self.new_stack = np.copy(self.raw_stack)
        self.impath = impath
        self.cell_list = []
        self.current = 0 # just using an index
        self.Nslice = self.raw_stack.shape[0] # number of slices 
        print("The number of slices:", self.Nslice)
        self.px_num = self.raw_stack.shape[1:] # the number of pixels in x and y for each slice
        print("The shape of images:", self.px_num)
        self.status = -1* np.ones(self.Nslice) # the status 
        self.sig = sig
        
        
        """
        status = -1: the slices are raw
        status = 0: current slice deblurred and aligned 
        status = 1: current slice deblurred but not aligned 
        """
    
    
    def image_high_trunc_inplane(self, nslice = None):
    # the idea comes from Nature-scientific reports,  3:2266.
    # im0: image 0 
    # method: filter method
    # ext: extension of the filter 
    # sig: width of the Gaussian filter
        sig = self.sig
        if (nslice is None):
            im0 = self.new_stack[self.current]
        else:
            im0 = self.new_stack[nslice]
            
            
        ifilt = filters.gaussian(im0, sigma=sig)
        iratio = im0/ifilt
        nmin = np.argmin(iratio) 
        gmin_ind = np.unravel_index(nmin, im0.shape) # global mininum of the index    
        sca =im0[gmin_ind]/(ifilt[gmin_ind])
        print(sca)
        im0 -= (ifilt*sca*0.98) # update the background-corrected image
        return im0




    def image_high_trunc_adjacent(self, wt = 0.40, sca = 3.0):
        sig = sca*self.sig # the Gaussian filter width of the adjacent slice 
        # Should this be done after the in-plane deblurring is accomplished, or before that? 
        if self.current > 0 and self.current < (self.Nslice-1):
            i_mid = self.raw_stack[self.current]
            i_up = self.raw_stack[self.current-1] # the previous slice 
            i_down = self.raw_stack[self.current+1] # the next slice 
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
            
        
        self.new_stack = np.copy(self.raw_stack)
        return self.new_stack
    
    def write_stack(self, n_apdx = None):
        if n_apdx is None:
            # overwrite the original file
            tifffunc.write_tiff(self.new_stack, self.impath)
        else: # add some appendix 
            tifffunc.write_tiff(self.new_stack, self.impath+n_apdx)
    
        

class Drift_correction(object):
    '''
    # updated on 08/02.  
    mfit: 0 --- linear correction
          1 --- Gaussian correction 
          2 --- nonlinear correction?  
    '''
    def __init__(self, raw_stack, mfit = 0):
        self.stack = raw_stack
        self.nslices = raw_stack.shape[0] # number of slices
        self.idim = np.array(raw_stack.shape[1:])
        self.mfit = mfit 
         
        # have a raw stack
   
   
    def pair_correct(self, im_ref = None, im_cor = None):
        # calculate the pixel shift between a pair of images (no correction is done at this moment!) 
        # no further argument tested.
        
        if(im_ref is None):
            im_ref = self.im_ref
        if(im_cor is None):
            im_cor = self.im_cor
        
        print("Image marks:")
        print(np.mean(im_ref), np.mean(im_cor))
        ft_ref = fftp.fft2(im_ref) # fft of the reference image 
        ft_cor = fftp.fft2(im_cor) 
        # here do the fftw-2d
        # some shift might be necessary to 
#          Cxy=ifft2(conj(FT_ref).*FT_shif);

        drift = self._shift_calculation(ft_ref,ft_cor)
        # shift back y first and x second 
        im_cor = np.roll(im_cor, -drift[0], axis = 0)
        im_cor = np.roll(im_cor, -drift[1], axis = 1)
        
        return im_cor
    
    def _shift_calculation(self, ft_ref,ft_cor):
         
        F_prod = np.conj(ft_ref)*ft_cor
        X_corr = fftp.ifft2(F_prod).astype('float')
        
        
        # findout the maximum of X_corr and fit to the Gaussian 
        Xmax = np.argmax(X_corr)
        nrow, ncol = np.unravel_index(Xmax, X_corr.shape)
        
        def res_ind(indi, hdim):
#             hdim: the dimension of the array. If the indi is larger than half of hdim, then indi is replaced by indi-hdim.
            if(indi>hdim/2):
                indo = indi - hdim
            else:
                indo = indi
            return indo
        
        dry=res_ind(nrow,self.idim[0])
        drx=res_ind(ncol,self.idim[1])
        
        
        if(self.mfit == 0):
            drift = [dry, drx]
            print("Drifted pixel:", drift)
            
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

            xrange = np.arange(Cnx-self.mfit, Cnx+ self.mfit+1) # Again, this "+1" is due to the subtle differences between matlab and python. 
            yrange = np.arange(Cny-self.mfit, Cny+ self.mfit+1)                
            corr_profile = X_corr[yrange,:][:,xrange]
            # next, let's determine the initial value of the arguments 
            
            popt = fitgaussian2D(corr_profile)
            cx = popt[1]
            cy = popt[2]
            
            print("center found at:", cx, cy)
            drift = [dry,  drx] + np.round([cy, cx]).astype('int64')-self.mfit


        print(drift)            
        return drift
            
    
    def drift_correct(self, offset = 1):
        # offset = m: start from the mth slice  
#         self.mfit = mfit
        im_ref = self.stack[offset]
        for ii in np.arange(offset+1,self.nslices):
            im_cor = np.copy(self.stack[ii])
            self.stack[ii]=self.pair_correct(im_ref,im_cor)
            im_ref = self.stack[ii]
            
        return self.stack



    
    