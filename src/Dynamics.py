"""
Created by Dan Xie on 08/15/2016 
Dynamics.py: takes the extracted cell information, calculate dynamics in it.
Updated by Dan on 08/17/2016 
Yay! Wire together, fire together. 
"""

import numpy as np
from scipy import linalg
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
            n_time, n_cell = TS_data.shape[:-1]
            self.coord = TS_data[0,:,:-1] # the coordinates of all the cells 
            self.n_cell = n_cell
            self.data_type = 'a'
            
        self.n_time = n_time
        self.fig_d = None
        self.sweet_list = np.array([]) # create an empty list to save the good cells. This really requires a GUI.
        
        
        
    def signal_profile_single(self, marker, rad = 20):
        """
        only works for nd_array data type.
        input: cell coordinate in the order of [y, x] in pixels
        rad: radius of selection
        output: The time_sercell_selectio 
        """
        yy = self.coord[:,0]
        xx = self.coord[:,1]
        yc, xc = marker
        
        r2 = (yy-yc)**2 + (xx-xc)**2
        
        c_select = (r2 <= rad*rad)  # the selected cells
        if(np.any(c_select)):
            f_select = self.ts_data[:,c_select, 2]
            return f_select
        else:
            print('No cell within the range.')
        # done with signal_profile_single
        
    
    def cell_show(self, dr = 6, ref_img = None):
        """
        A simple display of cell distributions.
        OK this kinda works. How to return the coordinates upon mouse clicking?
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

        
        
    