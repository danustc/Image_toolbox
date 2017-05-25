'''
Created on 04/30/2017 by Dan, the visualization tools of statistical learning results of the data.
Last update: 05/03/2017
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
        tx.annotate('PC '+ str(nr+1), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
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
            if ax.is_last_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('bottom')

            # plot the transposed half
            ax = axes[nc,nr]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.scatter(pc_col, pc_row, c = 'b', s = 5, cmap = plt.cm.spectral)

            if ax.is_last_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('top')
        # two-dimensional plot 

    return fig_pc


def PCA_trajectory_matrix(pc_data, dim_select = None):
    '''
    PCA representation of df/f trajectories.
    to be filled later.
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
        tx.annotate('PC '+ str(nr+1), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
        for nc in np.arange(nr):
            # iterate over columns
            pc_col= pc_data[:,dim_select[nc]]
            ax = axes[nr,nc]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.plot(pc_col,pc_row, c = 'g', linewidth = 2 )
            if ax.is_first_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('bottom')

            # plot the transposed half
            ax = axes[nc,nr]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.plot(pc_row, pc_col, c = 'g', linewidth = 2)

            if ax.is_last_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('top')
        # two-dimensional plot 

    return fig_pc


def pc_component_grid(V, npc = 3):
    '''
    visualize the first npcth principal components
    '''
    lx = 8
    NV, NP = V.shape # the number of components and the number of dimensions
    ndisplay = np.min([npc, NV])
    fig = plt.figure(figsize = (lx, ndisplay*lx/NV) ) # figuresize
    ax = fig.add_subplot(111)
    ax.imshow(V[:ndisplay]**2,cmap = 'Greens')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')

    return fig


def nature_style_dffplot(dff_data, dt = 0.8, sc_bar = 0.25):
    """
    Present delta F/F data in nature style
    """
    n_time, n_cell = dff_data.shape
    tt = np.arange(n_time)*dt

    tmark = -dt*10


    fig = plt.figure(figsize = (7,12))
    for ii in np.arange(n_cell):
        dff = dff_data[:,ii]
        ax = fig.add_subplot(n_cell,1, ii+1)
        ax.plot(tt, dff)
        ax.plot([tmark,tmark], [0, sc_bar], color = 'k', linewidth = 3)
        ax.set_xlim([-dt*20, tt[-1]])

        ax.set_ylim([-0.05, sc_bar*5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('time (s)', fontsize = 12)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0)

    return fig


# Raster plot, color coded
def dff_rasterplot(dff_ordered, dt = 0.5, fw = 7.0, tunit = 's'):
    '''
    dff_ordered: df_f ordered from most active to least active
    # rows:  # of time points
    # columns: # of cells
    Must be transposed before plotting.
    fw: figure width
    '''
    NT, NC = dff_ordered.shape
    # whether to display in the unit of 10 seconds or 1 min
    if(tnuit == 's'):
        time_tick = dt*np.arange(0, NT, 10)
    elif(tnuis == 'm'):
        time_tick = dt*np.arange(0, NT, 60)/60

    fig = plt.figure(figsize = (fw, fw*NC/NT))
    ax = fig.add_subplot(111)
    rshow = ax.imshow(dff_ordered.T, cmap = 'Greens', interpolation = 'None')
    ax.set_xticks(time_tick, fontsize = 12)
    cbar = fig.colorbar(rshow, ax = ax, extend = 'max', orientation = 'vertical', pad = 0.02)
    cbar.ax.tick_params(labelsize = 12)

    return fig



