import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk



def cluster_show(coords, vol_img = None, cm = 'viridis', ax = None):
    '''
    cluster_coords: list, each element stores coordinates of cells in a cluster, saved in x-y order.
    '''
    if ax is None:
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(111)

    if vol_img is None:
        pass
    else:
        NY, NX = vol_img.shape
        ax.imshow(vol_img, cmap = 'Greys_r', extent = [0, 0.295*NX, 0.295*NY, 0])

    ax.scatter(coords[:,0], coords[:,1], s = 9, c = coords[:,2], cmap = cm, vmin = 0, vmax = 100) # put the Z-information into d

    ax.axis('off')
    #return ax


def multi_cluster_show(cluster_ccs, vol_img, cm, layout = None):
    '''
    plot multiple clusters in an array specified by layout
    '''
    if layout is None:
        NR = 1
        NC = len(cluster_ccs)
    else:
        NR, NC = layout

    fig, axes = plt.subplots(nrows = NR, ncols = NC)
    fig.subplots_adjust(hspace = 0.05, wspace = 0.02)
    for ii in range(NR):
        for jj in range(NC):
            ax = axes[ii,jj]
            kk = ii*NR+jj+1
            try:
                cluster_show(cluster_ccs[kk-1], vol_img, cm, ax)
            except IndexError:
                print(kk, "Out of bound.")
                break # jump out of the loop

    plt.tight_layout()
    return fig


