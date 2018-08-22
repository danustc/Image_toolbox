'''
Created on 04/30/2017 by Dan, the visualization tools of statistical learning results of the data.
Last update: 07/12/2017
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
default_ccode = ['coral', 'teal', 'darkviolet', 'sienna', 'mediumblue', 'dimgrey']

def direct_dimplot(coef_data, ccode = default_ccode[0], lb_name = 'PC'):
    '''
    coef_data: NC*NP matrix
    '''
    NC, NP = coef_data.shape
    fig_dim, axes = plt.subplots(nrows = NP, ncols = NP, figsize = (2.1*NP, 2*NP))
    fig_dim.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for nr in range(NP):
        tx = axes[nr,nr]
        tx.xaxis.set_visible(False)
        tx.yaxis.set_visible(False)
        tx.annotate(lb_name + ' '+ str(nr+1), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
        pc_row = coef_data[:,nr]
        for nc in range(nr):
            pc_col = coef_data[:,nc]
            # iterate over columns
            ax = axes[nr,nc]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.scatter(pc_col,pc_row, c = ccode, s = 10)
            # done with color-coded group plot
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
            ax.scatter(pc_row,pc_col, c = ccode, s = 10)

            if ax.is_last_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('top')
    plt.tight_layout()
    return fig_dim


def cluster_dimplot(cp_data, cluster_indices, ccode = default_ccode, lb_name = 'IC'):
    '''
    cp_data: NT*NP matrix, the column represent different dimensions of features and the row represent different cells
    cluster_indices: list of lists, specifying different groups
    ccodes: colorcoding of clusters in the cluster_indices
    '''
    NT, NP = cp_data.shape
    n_cluster = len(cluster_indices)
    fig_pc, axes = plt.subplots(nrows = NP , ncols = NP, figsize = (2.1*NP, 2*NP))
    fig_pc.subplots_adjust(hspace = 0.03, wspace = 0.03)
    for nr in range(NP):
        #iterate over rows
        tx = axes[nr,nr]
        tx.xaxis.set_visible(False)
        tx.yaxis.set_visible(False)
        tx.annotate(lb_name + ' '+ str(nr+1), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
        for nc in range(nr):
            # iterate over columns
            ax = axes[nr,nc]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            for nf in range(n_cluster):
                pc_row = cp_data[cluster_indices[nf],nr]
                pc_col= cp_data[cluster_indices[nf],nc]
                ax.scatter(pc_col,pc_row, c = ccode[nf], s = 10)
            # done with color-coded group plot
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
            for nf in range(n_cluster):
                pc_row = cp_data[cluster_indices[nf],nr]
                pc_col= cp_data[cluster_indices[nf],nc]
                ax.scatter(pc_row,pc_col, c = ccode[nf], s = 10)

            if ax.is_last_col():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('top')
    plt.tight_layout()
    return fig_pc



def PCA_trajectory_matrix(cp_data, dim_select = None):
    '''
    PCA representation of df/f trajectories.
    to be filled later.
    '''
    if dim_select is None:
        dim_select = np.arange(np.max([3, cp_data.shape[1]]))

    ndim = len(dim_select) # the dimension of dim_select
    fig_pc, axes = plt.subplots(nrows = ndim, ncols = ndim, figsize = (2*ndim, 2*ndim))
    fig_pc.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for nr in range(ndim):
        #iterate over rows
        pc_row = cp_data[:,dim_select[nr]]
        tx = axes[nr,nr]
        tx.xaxis.set_visible(False)
        tx.yaxis.set_visible(False)
        tx.annotate('PC '+ str(nr+1), (0.5, 0.5),xycoords = 'axes fraction', ha = 'center', va = 'center', fontsize = 14)
        for nc in np.arange(nr):
            # iterate over columns
            pc_col= cp_data[:,dim_select[nc]]
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
    fig = plt.figure(figsize = (lx, ndisplay*lx/NP+0.5) ) # figuresize
    ax = fig.add_subplot(111)
    ax.imshow(V[:ndisplay]**2,cmap = 'Greens', extent = [1, NP, NV, 1], aspect = 'auto')
    ax.set_xticks([1,NP])
    ax.set_xticklabels(['Cell 1', 'Cell '+str(NP)])
    ax.set_yticks([1, NV])
    ax.set_yticklabels(['PC 1', 'PC '+str(ndisplay)])
    ax.set_aspect('equal')
    ax.tick_params(labelsize = 12)
    plt.tight_layout()
    return fig


def clusters_3d_distribution(ic_coeffs, cluster_indices, ccode = default_ccode, title = None):
    '''
    The 3D representation of the IC coefficients, clustered
    '''
    fig_3d = plt.figure(figsize = (5,4))
    ax = Axes3D(fig_3d, elev = 50, azim = 135)
    ic = 0
    for cluster in cluster_indices:
        ax.scatter(ic_coeffs[cluster,0], ic_coeffs[cluster,1], ic_coeffs[cluster,2], c = ccode[ic], s = 10)
        ic +=1

    ax.set_xlabel('ic 1', fontsize = 12)
    ax.set_ylabel('ic 2', fontsize = 12)
    ax.set_zlabel('ic 3', fontsize = 12)
    ax.dist = 12
    if title is not None:
        ax.set_title(title)
    return fig_3d



def ic_plot(ic_components, dt = 0.5, ccode = None, title = None):
    '''
    plot independent components
    '''
    NT, NC = ic_components.shape # number of time points and independent components
    fig, axes = plt.subplots(figsize = (8, 1.5*NC+0.5), nrows = NC, ncols = 1)
    for icon in range(NC):
        if ccode is None:
            cm = 'g'
        else:
            cm = ccode[icon]
        arow = axes[icon]
        arow.plot(dt*np.arange(NT), ic_components[:,icon],  c = cm, label = "ic_"+str(icon))
        arow.get_xaxis().set_visible(False)
        arow.get_yaxis().set_visible(False)
        arow.tick_params(labelsize = 12)
        arow.legend(['ic_'+str(icon+1)], loc = 'upper right', fontsize = 12)

    arow.get_xaxis().set_visible(True)
    arow.set_xlabel('Time (s)', fontsize = 12)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0)

    return fig


