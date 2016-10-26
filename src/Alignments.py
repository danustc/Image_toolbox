"""
Splitted from Background_correction.py on 10/23 by Dan.
Added: cross alignment
"""

import numpy as np
import scipy.fftpack as fftp
from numeric_funcs import fitgaussian2D


def res_ind(indi, hdim):
#   hdim: the dimension of the array. If the indi is larger than half of hdim, then indi is replaced by indi-hdim.
    if(indi>hdim/2):
        indo = indi - hdim
    else:
        indo = indi
    return indo
    # done with res_ind


def correlation_drift(im_ref, im_corr):
    """
    An independent function that can be used for drift correction. 
    """
    ny, nx = im_ref.shape # assume that the two frames have the same size. 
    
    ft_ref = fftp.fft2(im_ref)
    ft_corr = fftp.fft2(im_corr)
    







def cross_alignment(stack_ref, frame_al, z_step=1.0, z_al = 0.0, pre_align = True):
    """
    Align one or more slices into a stack.
    stack_ref: the reference stack 
    frame_al: the frame to be aligned. 
    z_step: the step of the reference stack 
    z_al: the z-coordinate of the to-be-aligned frame. 
    return: drift coordinates
    """
    nz = stack_ref.shape[0] # the number of slices 
    z_coordinates = np.arange(nz)* z_step # the z_coordinates of the 
    insert_ind = np.searchsorted(z_coordinates, z_al) # find where 
    
    
    
    
    
    
    





#-------------------------------------Class for drift correction -------------------------------

class Drift_correction(object):
    '''
    # updated on 08/30.  
    mfit: 0 --- linear correction
          1 --- Gaussian correction 
          2 --- nonlinear correction?  
    '''
    def __init__(self, stack, mfit = 0):
        """
        do not save the stack in the class.
        """
        self.stack = stack
        self.nslices = stack.shape[0] # number of slices
        self.idim = np.array(stack.shape[1:])
        self.mfit = mfit 
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
        return drift
            
    
    def drift_correct(self, offset = 1, ref_first = False, roll_back = False):
        """
        Update on 09/09: keep the drift list, so the cell coordinates might need be updated.
        """
        # offset = m: start from the mth slice  
#         self.mfit = mfit
        iref = offset
        im_ref = self.stack[iref]
        ft_ref = fftp.fft2(im_ref)
        drift_list = np.zeros([self.nslices, 2])
        
        if (ref_first == False):
            for icor in np.arange(offset+1,self.nslices):
                im_cor = self.stack[icor]
                ft_cor = fftp.fft2(im_cor)
                ft_ref = fftp.fft2(im_ref)
                drift = self._shift_calculation(ft_ref, ft_cor)
                if(roll_back):
                    im_cor = np.roll(im_cor, -drift[0], axis = 0)
                    im_cor = np.roll(im_cor, -drift[1], axis = 1)
                    self.stack[icor] = im_cor
                
                drift_list[icor] = drift
                im_ref = im_cor # reuse ft_ref
        else:
            # take the first slice as the reference
            for icor in np.arange(offset+1,self.nslices):
                im_cor = self.stack[icor]
                ft_cor = fftp.fft2(im_cor)
                drift = self._shift_calculation(ft_ref, ft_cor)
                if(roll_back):
                    im_cor = np.roll(im_cor, -drift[0], axis = 0)
                    im_cor = np.roll(im_cor, -drift[1], axis = 1)
                    self.stack[icor] = im_cor
                drift_list[icor] = drift
                # differs from the ref_first False case by the last statement
        self.drift_list = drift_list
#         print(drift_list)
    # done with stack drift correction
    
    
    def get_stack(self):
        """
        simply return the stack (corrected or not)
        """
        return self.stack

    
    def get_drift(self):
        """
        simply return the drift list 
        """
        return self.drift_list
#-------------------------------Done with drift correction part ----------------------------------------------------------


