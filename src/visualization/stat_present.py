'''
Created on 04/30/2017 by Dan, the visualization tools of statistical learning results of the data.
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCA_scatter_matrix(pc_data, dim_select = None):
    '''
    PCA_presentation of the pc_data.
    update: instead of 3D plot, generate a grid plot and make it a scatter matrix
    '''
    if dim_select is None:
        dim_select = np.arange(np.max(3, pc_data.shape[1]))

    ndim = len(dim_select) # the dimension of dim_select
    fig_pc, axes = plt.subplots(nrows = ndim, ncols = ndim, figsize = (2*ndim, 2*ndim))
    fig_pc.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for nr in range(ndim):
        #iterate over rows
        pc_row = pc_data[:,dim_select[nr]]
        tx = axes[nr,nr]
        tx.xaxis.set_visible(False)
        tx.yaxis.set_visible(False)
        tx.annotate('PC '+ str(nr), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
        for nc in np.arange(nr):
            # iterate over columns
            pc_col= pc_data[:,dim_select[nc]]
            ax = axes[nr,nc]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.scatter(pc_row,pc_col, c = 'b', s = 5, cmap = plt.cm.spectral)
            if ax.is_first_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('left')

            # plot the transposed half
            ax = axes[nc,nr]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.scatter(pc_col, pc_row, c = 'b', s = 5, cmap = plt.cm.spectral)

        # two-dimensional plot 

    return fig_pc


def PCA_trajectory(pc_data, dim_select = None):
    '''
    PCA representation of df/f trajectories.
    to be filled later.
    '''
    pass


