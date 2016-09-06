"""
Created by Dan Xie on 08/15/2016 
Dynamics.py: takes the extracted cell information, calculate dynamics in it.
Last update: 08/19/16: calculate all the df_f. Remove the function "sweet_build". 
Yay! Wire together, fire together. 
"""

import numpy as np
from scipy import fftpack
from df_f import dff_raw
import matplotlib.pyplot as plt
 

class Temporal_analysis(object):
    """
    Purpose: load one slice-all the time points
    No image operation, all based on .npz operations.   
    Recognize all the cell in the same position, extract its time train
    """

    def __init__(self, TS_data, dims, ref_im = None):
        """
        TS_data: the blob values of 
        """
       
        self.dims = dims# Is it OK to always keep the time series inside the memory? 
        
        
        if(type(TS_data) == np.ndarray):
            # OK! We got an python numpy array
            
            self.ref_im = np.zeros(dims)
            self.n_time, self.n_cell = TS_data.shape[:-1]
            
            self.coord = TS_data[0,:,:-1] # the coordinates of all the cells 
            self.ts_data = TS_data 
            self.data_type = 'a'
            
        else:
            # each frame has its own cell extractions
            self.data_type = 'd' 
            self.ts_data = TS_data['data']
            self.coord = TS_data['xy']
            self.n_time, self.n_cell = self.ts_data.shape[:2]
            
        if(ref_im is None):
            # create a reference image 
            self.ref_im = None
        else:
            self.ref_im = ref_im
            
            
        
        self.fig_d = None
        self.sweet_list = np.array([]) # create an empty list to save the good cells. This really requires a GUI.
        # done with initialization. This is kinda long.
        
        
    def dff_calculation(self, ft_width = 10): 
        """ 
        calculate df/f for all the identified cells.
        OK this works.
        """
        nc = self.n_cell
        dff = np.zeros([self.n_time, nc]) 
        f0 = np.zeros_like(dff)
        if(self.data_type == 'a'):
            f_raw = self.ts_data[:,:,2] # taking out the signal part 
        else:
            f_raw = self.ts_data
        
        for ic in np.arange(nc):
            # calculate the dff for each cell
            dff[:,ic], f0[:,ic] = dff_raw(f_raw[:,ic], ft_width)
        
        self.dff = dff 
        self.f_base = f0
        return dff
        # done with dff_calculation
        
        

    def firing_analysis(self, sfreq = 1.25, kfrac = 0.20, k_threshold = 0.80):
        """
        Update on 08/19: replacing the raw data with dff. 
        Analyzes the firing pattern of all the neurons fft-based. Feature the frequency components larger than kfrac* kmax 
        Because the fourier-transformed data is symmetric w.r.t. center, the final version is truncated to half
        Then, set a threshold of the frequency domain component. Any cell that have k-components above the threshold is selected 
        as active neurons, while others are discarded. 
        """
        # -------------Part 1: prepare the data in the fourier domain, set the criteria for cell selection
        N = self.n_time
        k_max = sfreq*0.5 # The maximum resolvable frequency (Nyquist frequency)
        dk = sfreq/N # the frequency resolution
        N_cut = (k_max*kfrac/dk).astype('uint16') # the cutoff frequency 
        
        signal_all = self.dff
        ft_signal = np.abs(fftpack.fft(signal_all, axis = 0)) # Fourier transform of the original data
        ft_signal = ft_signal[N_cut[0]:N_cut[1],:] # truncate the ft_data 
        ks = np.array([np.arange(N_cut[0], N_cut[1])*sfreq/N]) # the range of frequency 
        
        # -------------Part 2: select the neurons that fit the maximum
        # for any cell, if it totally falls below k_cut, then it is discarded.  
        k_floor = np.mean(np.min(ft_signal,axis = 0))
        k_max = np.max(ft_signal,axis = 0)
        k_mean = np.mean(ft_signal,axis = 0)
        k_ceil = np.mean(k_max) 
        k_cut = k_ceil*k_threshold+k_floor*(1.-k_threshold)
        
        k_ind = k_mean > k_cut # this is a very rough selection
        k_select = np.arange(self.n_cell)[k_ind]
        
        kpros = np.concatenate((ks.T, ft_signal[:,k_select]), axis = 1)
        
        return kpros, k_select
        
        
        
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
        
    
    def cell_show(self, dr = 6):
        """
        A simple display of cell distributions.
        OK this kinda works. How to return the coordinates upon mouse clicking?
        Update: have a saved cell-distribution
        """
        slice_0 = self.coord # taking out the first slice 
        if(self.fig_d is None):
            fig = plt.figure(figsize = (7,6))
        else:
            fig = self.fig_d    
            fig.clf()
        
        ax = fig.add_subplot(1,1,1)
        if(self.ref_im is not None):
            distr = ax.imshow(self.ref_im, cmap = 'Greys_r', interpolation = 'none')
        else:
            distr = ax.imshow(np.zeros(self.dims), cmap = 'Greys_r')
        
        for blob in slice_0:
            # plot all the blobs in the figure 
            y, x = blob
            c = plt.Circle((x, y), dr, color='g', linewidth=1, fill=True)
            ax.add_patch(c)

        ax.axis('off')

        self.distr = distr
        self.fig_d = fig
        
        return distr 
    
    
    def cell_mark(self, mark, tx, dr = 6.5, cl = 'r'):
        """
        mark the selected cells from the figure 
        """
        fig = self.fig_d
        ax = fig.get_axes()[0] 
        
        y, x = mark
        c = plt.Circle((x, y), dr+1, color=cl, linewidth=1, fill=True)
        ax.add_patch(c)
        ax.text(mark[1], mark[0]+20, tx, color = 'w')

        return fig
        # done with cell_mark
    
    def dff_save(self, dph):
        """
        save df/f data (raw) as well as the cell coordinates as an .npz file. 
        """
        data_struct = dict()
        data_struct['xy'] = self.coord
        data_struct['data'] = self.dff
        
        np.savez(dph, **data_struct)
        # done with dff_save
    
    
