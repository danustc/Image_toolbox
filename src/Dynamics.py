"""
Created by Dan Xie on 08/15/2016 
Dynamics.py: takes the extracted cell information, calculate dynamics in it.
Updated by Dan on 08/17/2016: try to insert GUI. This would be very messy. Should I use layered design like Ryan's inControl package? 
Yay! Wire together, fire together. 
"""

import numpy as np
from scipy import fftpack
from scipy.linalg import svd as SVD # import SVD algorithm
from numeric_funcs import circs_reconstruct
from graphic_funcs import coord_click


import matplotlib.pyplot as plt
 

class Temporal_analysis(object):
    """
    Purpose: load one slice-all the time points
    No image operation, all based on .npz operations.   
    Recognize all the cell in the same position, extract its time train
    
    """

    def __init__(self, TS_data, dims):
        """
        TS_data: the blob values of 
        """
        self.ts_data = TS_data 
        self.dims = dims# Is it OK to always keep the time series inside the memory? 
        if(type(TS_data) == type(dict())):
            # each frame has its own cell extractions
            self.data_type = 'd' 
            n_time = len(TS_data)
            self.n_cell = None # to be assigned later
            
            
        elif(type(TS_data) == np.ndarray):
            # OK! We got an python numpy array
            
            self.ref_img = np.zeros(dims)
            n_time, n_cell = TS_data.shape[:-1]
            self.coord = TS_data[0,:,:-1] # the coordinates of all the cells 
            self.n_cell = n_cell
            self.data_type = 'a'
            
        self.n_time = n_time
        self.fig_d = None
        self.sweet_list = np.array([]) # create an empty list to save the good cells. This really requires a GUI.
        # done with initialization. This is kinda long.
        
    
    def baseline(self, method = 'min', correct = False):
        """
        Creation date: 08/17
        This one calculates the baseline signal of each cell
        method options: 
        'min': simply select the minimum of each time-train. 
        'exp': fit the time-train with exponential function. 
        'avg': take the average of the time-train.
        ... Others to be added.
        """
        signal_all = self.ts_data[:,:,2] # take out the fluorescent signal 
        if(method == 'min'):
            self.base = np.min(signal_all, axis = 0) # minimum across time-train
        elif(method == 'exp'):
            # this is a bit tricky. To be filled up later.
            pass 
        else: 
            self.base = np.mean(signal_all, axis = 0)
            
        # done with baseline
        

    def firing_analysis(self, sfreq = 1.25, kfrac = 0.20):
        """
        Analyzes the firing pattern of all the neurons
        fft-based. Feature the frequency components larger than kfrac* kmax 
        """
        N = self.n_time
        signal_all = self.ts_data[:,:,2]
        ft_signal = np.abs(fftpack.fft(signal_all, axis = 0)) # Fourier transform of the original data 
        k_max = sfreq*0.5 # The maximum resolvable frequency (Nyquist frequency)
        ks = np.array([(np.arange(N)-0.5*N)*sfreq/N]) # the range of frequency 
        ks_data = np.concatenate((ks.T, ft_signal), axis = 1)
        
        return ks_data
        
        
        
    def signal_profile_single(self, marker, rad = 20, sel = 1):
        """
        only works for nd_array data type.
        input: cell coordinate in the order of [y, x] in pixels
        rad: radius of selection
        output: The time_sercell_selection 
        Update on 08/17: allows one or multiple selections. (defined by sel)
        """
        yy = self.coord[:,0]
        xx = self.coord[:,1]
        yc, xc = marker
        
        r2 = (yy-yc)**2 + (xx-xc)**2
        c_select = (r2 <= rad*rad)  # the selected cells
        if(np.any(c_select)):
            r_in = r2[c_select]
            n_sel = np.min([len(r_in), sel])
            st_int = np.argsort(r_in) # sort to have the 
            
            cflag_preselect = np.arange(self.n_cell)[c_select]
            cflag = cflag_preselect[st_int[:n_sel]]
            return cflag 

        else:
            print('No cell within the range.')
        # done with signal_profile_single
        
    
    def cell_show(self, dr = 6, ref_img = None):
        """
        A simple display of cell distributions.
        OK this kinda works. How to return the coordinates upon mouse clicking?
        Update: have a saved cell-distribution
        """
        slice_0 = self.ts_data[0, :, 0:-1] # taking out the first slice 
        if(self.fig_d is None):
            fig = plt.figure(figsize = (7,6))
        else:
            fig = self.fig_d    
            fig.clf()
        
        ax = fig.add_subplot(1,1,1)
        if(ref_img):
            distr = ax.imshow(ref_img, cmap = 'Greys')
        else:
            distr = ax.imshow(np.zeros(self.dims), cmap = 'Greys')
        
        for blob in slice_0:
            # plot all the blobs in the figure 
            y, x = blob
            c = plt.Circle((x, y), dr, color='g', linewidth=1, fill=True)
            ax.add_patch(c)

        self.distr = distr
        self.fig_d = fig
        
        return distr 
    
    
    def sweet_list_build(self):
        """
        Build a sweet list using the image 
        """
        if self.sweet_list: # sweet list is empty
            self.sweet_list = np.array([]) # clear it  
            
        distr = self.distr
        sweet_cc = coord_click(distr)
        plt.figure(self.fig_d.number)
        plt.show()
        sweet_list = sweet_cc.catch_values()
        self.sweet_list = sweet_list
        return sweet_list
    
        
    
    def feature_extract(self, t_level = 3):
        """
        extract the features of the dynamics, up to the t_level th singular value
        """
        mtx = self.set_dynamics
        U, s, V = SVD(mtx) # calculate SVD
        
        U_principal = U[:t_level, :] # the first t_level rows
        V_principal = V[:, :t_level] # the first t_level columns
        s_principal = s[:t_level] # the first t_level singular values
        
        # next, let's do some projection 
        C_principal = np.dot(U_principal, np.diag(s_principal))
        
        return C_principal, V_principal # return the coefficients and singular vectors.

        
        
    