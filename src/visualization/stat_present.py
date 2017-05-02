'''
Created on 04/30/2017 by Dan, the visualization tools of statistical learning results of the data.
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCA_scatter(pc_data, dim_select = None):
    '''
    PCA_presentation of the pc_data.
    '''
    fig_pc = plt.figure(figsize = (7,6))
    if dim_select is None:
        dim_select = np.arange(np.max(3, pc_data.shape[1]))
    if(len(dim_select) ==2):
        # two-dimensional plot 
        ax= fig_pc.add_subplot(111)
        ax.scatter(pc_data[:,dim_select[0]], pc_data[:,dim_select[1]], c = 'b', s = 11, cmap = plt.cm.spectral)
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ax.set_xlabel('PC'+str(dim_select[0]+1), fontsize = 14)
        ax.set_ylabel('PC'+str(dim_select[1]+1), fontsize = 14)

    else:
        # three-dimensional plot
        ax = Axes3D(fig_pc,elev = 45, azim = 135)
        ax.scatter(pc_data[:,dim_select[0]], pc_data[:, dim_select[1]], pc_data[:,dim_select[2]], cmap = plt.cm.spectral)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('PC'+str(dim_select[0]+1), fontsize = 14)
        ax.set_ylabel('PC'+str(dim_select[1]+1), fontsize = 14)
        ax.set_zlabel('PC'+str(dim_select[2]+1), fontsize = 14)



    return fig_pc


def PCA_trajectory(pc_data, dim_select = None):
    '''
    PCA representation of df/f trajectories.
    to be filled later.
    '''
    pass


