"""
This is df/f calculation based on the paper: Nature protocols, 6, 28–35, 2011
Created by Dan on 08/18/15
"""

import numpy as np
from numeric_funcs import smooth_lpf
import matplotlib.pyplot as plt

def min_window(shit_data, wd_width = 10):
    """
    Calculate the baseline  
    Very awkward 
    """
    f0 = np.zeros_like(shit_data)
    f0[0:wd_width] = np.min(shit_data[0:wd_width])
    N = len(f0)
    for ii in np.arange(wd_width, N):
        f0[ii] = np.min(shit_data[ii-wd_width:ii])
    return f0
# this is a good baseline calculation. 


def dff(shit_data, ft_width):
    """
    calculate df_f for shit_sig.
    """
    s_filt = smooth_lpf(shit_data, ft_width)[1]
    
    f_base = min_window(s_filt, 3*ft_width)
    dff_raw = (shit_data-f_base)/f_base
    
    return dff_raw
    

def nature_style_dffplot(dff_data, dt = 0.8, sc_bar = 0.25):
    """
    Present delta F/F data in nature style 
    """
    n_time, n_cell = dff_data.shape
    tt = np.arange(n_time)*dt
    
    tmark = -dt*10
    
    
    fig = plt.figure(figsize = (7,5))
    for ii in np.arange(n_cell):
        dff = dff_data[:,ii]
        ax = fig.add_subplot(n_cell,1, ii+1)
        ax.plot(tt, dff)
        ax.plot([tmark,tmark], [0, sc_bar], color = 'k', linewidth = 3)
        ax.set_xlim([-dt*20, tt[-1]])

        ax.set_ylim([-0.25, 1.25])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('time (s)', fontsize = 12)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0)
#     
 
    return fig
        
    
    
    



    
    
