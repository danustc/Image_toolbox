# ----------- Author: Dan Xie ------------------------
# ------------Last update: 07/22/2016 --------------------------
"""
This file deblurrs the in-plane image and deblurrs each image from its adjacent stacks 
"""

import tifffunc
import numpy as np
from skimage import filters
import scipy.fftpack as fftp
from vtk.util.colors import indian_red

class Preprocess(object):
    def __init__(self, impath, sig = 25):
        # sig is the width of Gaussian filter 
        self.raw_stack = tifffunc.read_tiff(impath).astype('float') # load a raw image and convert to float type 
        self.cell_list = []
        self.current = 0 # just using an index
        self.new_stack = np.copy(self.raw_stack)
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
            im0 = self.raw_stack[self.current]
        else:
            im0 = self.raw_stack[nslice]
            
            
        ifilt = filters.gaussian_filter(im0, sigma=sig)
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
        

class Drift_correction(object):
    # updated on 08/01.    
    def __init__(self, raw_stack):
        self.stack = raw_stack
        self.nslices = raw_stack.shape[0] # number of slices 
        # have a raw stack
   
   
    def __pair_correct__(self,im1, im2):
        # correct a pair of images 
        ft_im1 = fftp.fft2(im1) # fft of the reference image 
        ft_im2 = fftp.fft2(im2) 
        # here do the fftw-2d
        # some shift might be necessary to 
#          Cxy=ifft2(conj(FT_ref).*FT_shif);
        idim = im1.shape
        F_prod = np.conj(ft_im1)*ft_im2
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
        
        dry=res_ind(nrow,idim[0])
        drx=res_ind(ncol,idim[1])
    

        return [nrow, ncol]   
        
    def drift_correct_linear(self):
        im_ref = self.stack[0]
        for ii in np.arange(1,self.nslices):
            im_corr = self.stack[ii]
            self.__pair_correct__(im_ref, im_corr)
            im_ref = im_corr
        
    
    def drift_correct_gaussian(self):
        pass    
        



""" below are some shared functions
"""        
def circ_shift(matr, nshifts, overwrite = True):
    # matr: the matrix that would be shifted
    # nshifts: the number of pixels that should be shifted in each dimension
    if overwrite == False:
        m_shift = np.copy(matr)
    else:
        m_shift = matr
        
    
    
    for ii in len(nshifts):
        ns = nshifts[ii]
        print(ns)
#         shift(matr, -ns)
    return m_shift    
         
        


    
    